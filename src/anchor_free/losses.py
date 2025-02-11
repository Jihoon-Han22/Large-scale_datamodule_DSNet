import torch
from torch.nn import functional as F

def calc_cls_loss(pred: torch.Tensor,
                  test: torch.Tensor,
                  mask,
                  kind: str = 'focal') -> torch.Tensor:
    """Compute classification loss on both positive and negative samples.

    :param pred: Predicted class. Sized [N, S].
    :param test: Class label where 1 marks positive, -1 marks negative, and 0
        marks ignored. Sized [N, S].
    :param kind: Loss type. Choose from (focal, cross-entropy).
    :return: Scalar loss value.
    """


    # print("pred")
    # print(pred)
    # print(pred.shape)

    # print("test")
    # print(test)
    # print(test.shape)
    if (not isinstance(pred, torch.Tensor)):
        pred = torch.from_numpy(pred)
    test = test.type(torch.long)

    # print("AFTER IF CASE")
    # print("pred")
    # print(pred)
    # print(pred.shape)

    # print("test")
    # print(test)
    # print(test.shape)
    num_pos = torch.sum(test, axis=1)
    # print("num_pos")
    # print(num_pos)

    # pred = pred.unsqueeze(-1)
    pred = torch.cat([1 - pred, pred], dim=-1)
    # print("pred")
    # print(pred)
    # print(pred.shape)

    if kind == 'focal':
        loss = focal_loss(pred, test, mask, reduction='sum')
    elif kind == 'cross-entropy':
        loss = F.nll_loss(pred.log(), test)
    else:
        raise ValueError(f'Invalid loss type {kind}')

    # print("loss")
    # print(loss)
    # loss = loss / num_pos
    loss = torch.div(loss, num_pos)
    # print("loss")
    # print(loss)
    return torch.mean(loss)


def iou_offset(offset_a: torch.Tensor,
               offset_b: torch.Tensor,
               eps: float = 1e-8
               ) -> torch.Tensor:
    """Compute IoU offsets between multiple offset pairs.

    :param offset_a: Offsets of N positions. Sized [N, 2].
    :param offset_b: Offsets of N positions. Sized [N, 2].
    :param eps: Small floating value to prevent division by zero.
    :return: IoU values of N positions. Sized [N].
    """
    left_a, right_a = offset_a[:, 0], offset_a[:, 1]
    left_b, right_b = offset_b[:, 0], offset_b[:, 1]

    length_a = left_a + right_a
    length_b = left_b + right_b

    intersect = torch.min(left_a, left_b) + torch.min(right_a, right_b)
    intersect[intersect < 0] = 0
    union = length_a + length_b - intersect
    union[union <= 0] = eps

    iou = intersect / union
    return iou


def calc_loc_loss(pred_loc: torch.Tensor,
                  test_loc: torch.Tensor,
                  cls_label: torch.Tensor,
                  kind: str = 'soft-iou',
                  eps: float = 1e-8
                  ) -> torch.Tensor:
    """Compute soft IoU loss for regression only on positive samples.

    :param pred_loc: Predicted offsets. Sized [N, 2].
    :param test_loc: Ground truth offsets. Sized [N, 2].
    :param cls_label: Class label specifying positive samples.
    :param kind: Loss type. Choose from (soft-iou, smooth-l1).
    :param eps: Small floating value to prevent division by zero.
    :return: Scalar loss value.
    """

    # print("pred_loc")
    # print(pred_loc)
    # print(pred_loc.shape)

    # print("test_loc")
    # print(test_loc)
    # print(test_loc.shape)
    cls_label = cls_label.type(torch.bool)
    # print("cls_label")
    # print(cls_label)
    # print(cls_label.shape)
    pred_loc = pred_loc[cls_label]
    test_loc = test_loc[cls_label]
    
    if kind == 'soft-iou':
        iou = iou_offset(pred_loc, test_loc)
        loss = -torch.log(iou + eps).mean()
    elif kind == 'smooth-l1':
        loss = F.smooth_l1_loss(pred_loc, test_loc)
    else:
        raise ValueError(f'Invalid loss type {kind}')

    return loss


def calc_ctr_loss(pred, test, pos_mask):
    pos_mask = pos_mask.type(torch.bool)

    pred = pred[pos_mask]
    test = test[pos_mask]

    # print("pred")
    # print(pred)
    # print(pred.shape)
    
    pred = pred.squeeze(1)

    # print("pred")
    # print(pred)
    # print(pred.shape)
    # print("test")
    # print(test)
    # print(test.shape)
    loss = F.binary_cross_entropy(pred, test)
    return loss


def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form.

    :param labels: Class labels. Sized [N].
    :param num_classes: Number of classes.
    :return: One-hot encoded labels. sized [N, #classes].
    """
    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]


def focal_loss(x: torch.Tensor,
               y: torch.Tensor,
               mask, 
               alpha: float = 0.25,
               gamma: float = 2,
               reduction: str = 'sum'
               ) -> torch.Tensor:
    """Compute focal loss for binary classification.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    :param x: Predicted confidence. Sized [N, D].
    :param y: Ground truth label. Sized [N].
    :param alpha: Alpha parameter in focal loss.
    :param gamma: Gamma parameter in focal loss.
    :param reduction: Aggregation type. Choose from (sum, mean, none).
    :return: Scalar loss value.
    """

    # print("x.shape")
    # print(x.shape)
    B, d, num_classes = x.shape

    # print(y)
    # print("y.shape")
    # print(y.shape)
    t = one_hot_embedding(y, num_classes)
    # print(t)
    # print("t.shape")
    # print(t.shape)

    f1_results = []
    for b in range(B):
        # print("mask")
        # print(mask.shape)
        # print("mask[b]")
        # print(mask[b].shape)
        # print("num_classes")
        # print(num_classes)
        # print("d")
        # print(d)
        if mask != None:
            maski = mask[b].float().expand(num_classes, d).transpose(1, 0)
        x_one_batch = x[b]
        t_one_batch = t[b]
        # print("x_one_batch")
        # print(x_one_batch)
        # print(x_one_batch.shape)

        # print("t_one_batch")
        # print(t_one_batch)
        # print(t_one_batch.shape)
        # p_t = p if t > 0 else 1-p
        p_t = x_one_batch * t_one_batch + (1 - x_one_batch) * (1 - t_one_batch)

        # print("p_t")
        # print(p_t)
        # print(p_t.shape)
        # p_t += 0.01
        # print("log(p_t)")
        # print(p_t.log())
        # print((p_t.log()).shape)
        # alpha_t = alpha if t > 0 else 1-alpha
        alpha_t = alpha * t_one_batch + (1 - alpha) * (1 - t_one_batch)

        # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.log()
        if mask != None:
            fl = torch.mul(fl, maski)
        # print("fl")
        # print(fl)
        # print(fl.shape)

        if reduction == 'sum':
            fl = fl.sum()
        elif reduction == 'mean':
            fl = fl.mean()
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f'Invalid reduction mode {reduction}')
        f1_results.append(fl)
        # print("f1_results")
        # print(f1_results)
    return torch.stack(f1_results)


def focal_loss_with_logits(x, y, reduction='sum'):
    """Compute focal loss with logits input"""
    return focal_loss(x.sigmoid(), y, reduction=reduction)
