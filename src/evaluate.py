import logging
from pathlib import Path

import numpy as np
import torch

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

# ADDED
from mrsum_model.mrsum_dataset import MrSumDataset, BatchCollator
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from mrsum_model.utils.evaluation_metrics import evaluate_summary
from mrsum_model.utils.generate_summary import generate_summary
from mrsum_model.utils.evaluate_map import generate_mrsum_seg_scores, top50_summary, top15_summary
from anchor_free.losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss
from anchor_free import anchor_free_helper

logger = logging.getLogger()


def evaluate(model, data_loader, nms_thresh, device, args, epoch):
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity')

    dataloader = iter(data_loader)

    fscore_history = []
    map50_history = []
    map15_history = []

    cls_loss_history = []
    loc_loss_history = []
    ctr_loss_history = []
    loss_history = []
    with torch.no_grad():
        for data in dataloader:
            video_name = data['video_name']
            seq = data['features']
            gtscore = data['gtscore']

            gt_summary = np.array(data['gt_summary'][0])
            cps = data['change_points']
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg']
            picks = data['picks']
            mask = None

            ## Computing loss
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, cps, n_frames, nfps, picks)
            target = keyshot_summ

            gtscore = data['gtscore']
            # print("seq")
            # print(seq)
            # print(seq.shape)
            # seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            # seq = seq.squeeze(0)

            cls_label = target
            loc_label = anchor_free_helper.get_loc_label(target)
            ctr_label = anchor_free_helper.get_ctr_label(target, loc_label)

            cls_label = torch.tensor(cls_label, dtype=torch.float32)
            loc_label = torch.tensor(loc_label, dtype=torch.float32)
            ctr_label = torch.tensor(ctr_label, dtype=torch.float32)


            ## Computing fscore
            # seq_torch = seq
            seq_torch = torch.from_numpy(np.array(seq)).to(device)
            _, seq_len,  _ = seq_torch.shape
            pred_cls, pred_loc, pred_ctr, pred_bboxes = model.predict(seq_torch)
            pred_cls = pred_cls.squeeze(0)
            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
            data_pred_cls, data_pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
            pred_summ, pred_score = vsumm_helper.bbox2summary(
                seq_len, data_pred_cls, data_pred_bboxes, cps, n_frames, nfps, picks)
            
            pred_summ = pred_summ[0]
            fscore= evaluate_summary(pred_summ, gt_summary, eval_method='avg')
            fscore_history.append(fscore)

            ## Computing loss
            pred_cls = torch.from_numpy(pred_cls).unsqueeze(0)
            pred_loc = torch.from_numpy(pred_loc).unsqueeze(0)

            # print("cls_label")
            # print(cls_label)
            # print(cls_label.shape)

            # print("pred_cls")
            # print(pred_cls)
            # print(pred_cls.shape)
            cls_loss = calc_cls_loss(pred_cls, cls_label, mask, args.cls_loss)
            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label,
                                     args.reg_loss)
            
            pred_ctr = pred_ctr.cpu()
            ctr_loss = calc_ctr_loss(pred_ctr, ctr_label, cls_label)

            loss = args.lambda_cls * cls_loss + args.lambda_reg * loc_loss + args.lambda_ctr * ctr_loss
            
            cls_loss_history.append(cls_loss)
            loc_loss_history.append(loc_loss)
            ctr_loss_history.append(ctr_loss)
            loss_history.append(loss)

            ### Highlight Detection Metric ###
            gt_seg_score = generate_mrsum_seg_scores(gtscore.squeeze(0), uniform_clip=5)
            gt_top50_summary = top50_summary(gt_seg_score)
            gt_top15_summary = top15_summary(gt_seg_score)
            
            highlight_seg_machine_score = generate_mrsum_seg_scores(pred_score.squeeze(0), uniform_clip=5)
            highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (torch.exp(highlight_seg_machine_score).sum() + 1e-7)
            
            clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu()
            clone_machine_summary = clone_machine_summary.numpy()
            aP50 = average_precision_score(gt_top50_summary, clone_machine_summary)
            aP15 = average_precision_score(gt_top15_summary, clone_machine_summary)
            map50_history.append(aP50)
            map15_history.append(aP15)
        mean_cls_loss = np.mean(np.array(cls_loss_history))
        mean_loc_loss = np.mean(np.array(loc_loss_history))
        mean_ctr_loss = np.mean(np.array(ctr_loss_history))
        mean_loss = np.mean(np.array(loss_history))

        final_f_score = np.mean(fscore_history)
        final_map50 = np.mean(map50_history)
        final_map15 = np.mean(map15_history)
        stats.update(fscore=final_f_score)
        return final_f_score, final_map50, final_map15, mean_cls_loss, mean_loc_loss, mean_ctr_loss, mean_loss


def main():
    print("start")
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))
    print("load_model")
    model = get_model(args.model, **vars(args))
    print("model.eval_ start")
    model = model.eval().to(args.device)
    print("model.eval done")
    stats = data_helper.AverageMeter('fscore', 'diversity')
    
    ckpt_path = data_helper.get_ckpt_path(args.model_dir, args)
    state_dict = torch.load(str(ckpt_path),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    test_dataset = MrSumDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    fscore, diversity = evaluate(model, test_loader, args.nms_thresh, args.device)
    stats.update(fscore=fscore, diversity=diversity)
    logger.info(f'{args.tag}: Test F-score: {fscore:.4f}')
    print(f'{args.tag}: Test F-score: {fscore:.4f}')


if __name__ == '__main__':
    main()
