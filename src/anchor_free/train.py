import logging

import torch

from anchor_free import anchor_free_helper
from anchor_free.dsnet_af import DSNetAF
from anchor_free.losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss
from evaluate import evaluate
from helpers import data_helper, vsumm_helper

# ADDED
from tqdm import tqdm
import os
from mrsum_model.utils.utils import TensorboardWriter
import numpy as np

logger = logging.getLogger()


def train(args, train_loader, val_loader, test_loader, save_path, log_dir):
    model = DSNetAF(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head)
    model = model.to(args.device)

    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                 weight_decay=args.weight_decay)
    writer = TensorboardWriter(str(log_dir))

    max_val_fscore = -1

    for epoch in range(args.max_epoch):
        print("[Epoch: {0:6}]".format(str(epoch)+"/"+str(args.max_epoch)))
        model.train()
        stats = data_helper.AverageMeter('loss', 'cls_loss', 'loc_loss',
                                         'ctr_loss')
        num_batches = int(len(train_loader))
        iterator = iter(train_loader)


        cls_loss_history = []
        loc_loss_history = []
        ctr_loss_history = []
        loss_history = []

        for _ in tqdm(range(num_batches)):
            data = next(iterator)
            video_name = data['video_name']
            seq = data['features']
            gtscore = data['gtscore']

            gt_summary = data['gt_summary'][0]
            mask = data['mask'].to(args.device)
            cps = data['change_points']
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg']
            picks = data['picks']

            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, cps, n_frames, nfps, picks)
            target = keyshot_summ
            # target = vsumm_helper.downsample_summ(keyshot_summ)

            # print("cps")
            # print(cps)
            # print(len(cps))
            # print("target")
            # print(target)
            # print(target.shape)
            # print("gt_summary")
            # print(gt_summary)
            # print(gt_summary.shape)

            # if not target.any():
            #     continue

            gtscore = data['gtscore'].to(args.device)
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            seq = seq.squeeze(0)
            cls_label = target
            loc_label = anchor_free_helper.get_loc_label(target)
            ctr_label = anchor_free_helper.get_ctr_label(target, loc_label)
            # print("cls_label")
            # print(cls_label)
            # print(cls_label.shape)
            # print("loc_label")
            # print(loc_label)
            # print(np.shape(loc_label))
            # print("ctr_label")
            # print(ctr_label)
            # print(np.shape(ctr_label))


            pred_cls, pred_loc, pred_ctr = model(seq, mask)
            
            # print("pred_cls")
            # print(pred_cls)
            # print(pred_cls.shape)
            # print("pred_loc")
            # print(pred_loc)
            # print(pred_loc.shape)
            # print("pred_ctr")
            # print(pred_ctr)
            # print(pred_ctr.shape)
            # print("cls_label")
            # print(cls_label)
            # print(cls_label.shape)
            cls_label = torch.tensor(cls_label, dtype=torch.float32).to(args.device)
            loc_label = torch.tensor(loc_label, dtype=torch.float32).to(args.device)
            ctr_label = torch.tensor(ctr_label, dtype=torch.float32).to(args.device)

            cls_loss = calc_cls_loss(pred_cls, cls_label, mask, args.cls_loss)
            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label,
                                     args.reg_loss)
            ctr_loss = calc_ctr_loss(pred_ctr, ctr_label, cls_label)

            loss = args.lambda_cls * cls_loss + args.lambda_reg * loc_loss + args.lambda_ctr * ctr_loss

            optimizer.zero_grad()
            loss.backward()

            cls_loss_history.append(cls_loss)
            loc_loss_history.append(loc_loss)
            ctr_loss_history.append(ctr_loss)
            loss_history.append(loss)

            optimizer.step()

            # stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
            #              loc_loss=loc_loss.item(), ctr_loss=ctr_loss.item())

        # print("cls_loss_history")
        # print(cls_loss_history)
        # print(cls_loss_history.shape)
        mean_cls_loss = torch.mean(torch.stack(cls_loss_history))
        mean_loc_loss = torch.mean(torch.stack(loc_loss_history))
        mean_ctr_loss = torch.mean(torch.stack(ctr_loss_history))
        mean_loss = torch.mean(torch.stack(loss_history))

        writer.update_loss(mean_cls_loss, epoch, 'train/cls_loss_epoch')
        writer.update_loss(mean_loc_loss, epoch, 'train/loc_loss_epoch')
        writer.update_loss(mean_ctr_loss, epoch, 'train/ctr_loss_epoch')
        writer.update_loss(mean_loss, epoch, 'train/loss_epoch')
        
        writer.update_loss(optimizer.param_groups[-1]['lr'], epoch, 'current_lr')

        print("one epoch done")
        logger.info(f'[Train INFO] Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {mean_cls_loss:.4f}/{mean_loc_loss:.4f}/{mean_ctr_loss:.4f}/{mean_loss:.4f} ')
        val_fscore, val_map50, val_map15, val_cls_loss, val_loc_loss, val_ctr_loss, val_loss = evaluate(model, val_loader, args.nms_thresh, args.device, args, epoch)

        writer.update_loss(val_fscore, epoch, 'val/fscore_epoch')
        writer.update_loss(val_cls_loss, epoch, 'val/cls_loss_epoch')
        writer.update_loss(val_loc_loss, epoch, 'val/loc_loss_epoch')
        writer.update_loss(val_ctr_loss, epoch, 'val/ctr_loss_epoch')
        writer.update_loss(val_loss, epoch, 'val/loss_epoch')

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        print("Vals INFO")
        logger.info(f'[Vals INFO] Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {val_cls_loss:.4f}/{val_loc_loss:.4f}/{val_ctr_loss:.4f}/{val_loss:.4f} '
                    f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}')
    
    state_dict = torch.load(str(save_path),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    test_fscore, test_map50, test_map15, _, _, _, _ = evaluate(model, test_loader, args.nms_thresh, args.device, args, 0)
    print("------------------------------------------------------")
    print(f"   TEST RESULT on {save_path}: ")
    print('   TEST MRSum F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(test_fscore, test_map50, test_map15))
    print("------------------------------------------------------")
    
    f = open(os.path.join(args.model_dir, 'results.txt'), 'a')
    f.write("Testing on Model " + str(save_path) + '\n')
    f.write('Test F-score ' + str(test_fscore) + '\n')
    f.write('Test MAP50   ' + str(test_map50) + '\n')
    f.write('Test MAP15   ' + str(test_map15) + '\n\n')
    f.flush()
    
    
    
    return test_fscore, test_map50, test_map15
