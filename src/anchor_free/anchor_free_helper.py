import numpy as np

from helpers import bbox_helper


def get_loc_label(target: np.ndarray) -> np.ndarray:
    """Generate location offset label from ground truth summary.

    :param target: Ground truth summary. Sized [N].
    :return: Location offset label in LR format. Sized [N, 2].
    """
    # seq_len stores a list of length of sequences
    seq_len = []
    for b in range(len(target)):
        seq_len.append(target[b].size)

    bboxes = bbox_helper.seq2bbox(target)
    offsets = bbox2offset(bboxes, seq_len)

    return offsets


def get_ctr_label(target: np.ndarray,
                  offset: np.ndarray,
                  eps: float = 1e-8
                  ) -> np.ndarray:
    """Generate centerness label for ground truth summary.

    :param target: Ground truth summary. Sized [N].
    :param offset: LR offset corresponding to target. Sized [N, 2].
    :param eps: Small floating value to prevent division by zero.
    :return: Centerness label. Sized [N].
    """
    B = len(target)
    # print("B: ", B)

    result_ctr_label = []
    for b in range(B):
        target_len = target[b].size
        targ = target[b]
        targ = np.asarray(targ, dtype=np.bool)
        # print("targ")
        # print(targ)
        ctr_label = np.zeros(target_len, dtype=np.float32)
        # print("ctr_label")
        # print(ctr_label)
        # print("offset")
        # print(offset[b])
        offset_left, offset_right = offset[b][targ, 0], offset[b][targ, 1]
        # print("offset_left")
        # print(offset_left)
        # print("offset_right")
        # print(offset_right)
        ctr_label[targ] = np.minimum(offset_left, offset_right) / (
            np.maximum(offset_left, offset_right) + eps)
        result_ctr_label.append(ctr_label)
    return result_ctr_label


def bbox2offset(bboxes: np.ndarray, seq_len) -> np.ndarray:
    """Convert LR bounding boxes to LR offsets.

    :param bboxes: LR bounding boxes.
    :param seq_len: Sequence length N.
    :return: LR offsets. Sized [N, 2].
    """
    result_offsets = []
    B = len(bboxes)

    for b in range(B):
        pos_idx = np.arange(seq_len[b], dtype=np.float32)
        offsets = np.zeros((seq_len[b], 2), dtype=np.float32)

        for lo, hi in bboxes[b]:
            bbox_pos = pos_idx[lo:hi]
            offsets[lo:hi] = np.vstack((bbox_pos - lo, hi - 1 - bbox_pos)).T
        result_offsets.append(offsets)
    return result_offsets


def offset2bbox(offsets: np.ndarray) -> np.ndarray:
    """Convert LR offsets to bounding boxes.

    :param offsets: LR offsets. Sized [N, 2].
    :return: Bounding boxes corresponding to offsets. Sized [N, 2].
    """
    # print("offsets")
    # print(offsets)
    # print(offsets.shape)

    offset_left, offset_right = offsets[:, 0], offsets[:, 1]
    seq_len, _ = offsets.shape
    indices = np.arange(seq_len)
    bbox_left = indices - offset_left
    bbox_right = indices + offset_right + 1
    bboxes = np.vstack((bbox_left, bbox_right)).T
    # print("inside bboxes")
    # print(bboxes)
    # print(bboxes.shape)
    return bboxes
