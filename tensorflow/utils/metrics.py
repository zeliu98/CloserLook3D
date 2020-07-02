import numpy as np
from sklearn.metrics import confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def classification_metrics(preds, targets, num_classes):
    seen_class = [0.0 for _ in range(num_classes)]
    correct_class = [0.0 for _ in range(num_classes)]
    preds = np.argmax(preds, -1)
    correct = np.sum(preds == targets)
    seen = preds.shape[0]
    for l in range(num_classes):
        seen_class[l] += np.sum(targets == l)
        correct_class[l] += (np.sum((preds == l) & (targets == l)))
    acc = 1.0 * correct / seen
    avg_class_acc = np.mean(np.array(correct_class) / np.array(seen_class))
    return acc, avg_class_acc


def partnet_metrics(num_classes, num_parts, objects, preds, targets):
    shape_iou_tot = [0.0] * num_classes
    shape_iou_cnt = [0] * num_classes
    part_intersect = [np.zeros((num_parts[o_l]), dtype=np.float32) for o_l in range(num_classes)]
    part_union = [np.zeros((num_parts[o_l]), dtype=np.float32) + 1e-6 for o_l in range(num_classes)]

    for obj, cur_pred, cur_gt in zip(objects, preds, targets):
        cur_num_parts = num_parts[obj]
        cur_pred = np.argmax(cur_pred[:, 1:], axis=-1) + 1
        cur_pred[cur_gt == 0] = 0
        cur_shape_iou_tot = 0.0
        cur_shape_iou_cnt = 0
        for j in range(1, cur_num_parts):
            cur_gt_mask = (cur_gt == j)
            cur_pred_mask = (cur_pred == j)

            has_gt = (np.sum(cur_gt_mask) > 0)
            has_pred = (np.sum(cur_pred_mask) > 0)

            if has_gt or has_pred:
                intersect = np.sum(cur_gt_mask & cur_pred_mask)
                union = np.sum(cur_gt_mask | cur_pred_mask)
                iou = intersect / union

                cur_shape_iou_tot += iou
                cur_shape_iou_cnt += 1

                part_intersect[obj][j] += intersect
                part_union[obj][j] += union
        if cur_shape_iou_cnt > 0:
            cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt
            shape_iou_tot[obj] += cur_shape_miou
            shape_iou_cnt[obj] += 1

    msIoU = [shape_iou_tot[o_l] / shape_iou_cnt[o_l] for o_l in range(num_classes)]
    part_iou = [np.divide(part_intersect[o_l][1:], part_union[o_l][1:]) for o_l in range(num_classes)]
    mpIoU = [np.mean(part_iou[o_l]) for o_l in range(num_classes)]

    # Print instance mean
    mmsIoU = np.mean(np.array(msIoU))
    mmpIoU = np.mean(mpIoU)

    return msIoU, mpIoU, mmsIoU, mmpIoU


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU


def s3dis_subset_metrics(dataset, predictions, targets, val_proportions):
    # Confusions for subparts of validation set
    Confs = np.zeros((len(predictions), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for i, (probs, truth) in enumerate(zip(predictions, targets)):
        # Predicted labels
        preds = dataset.label_values[np.argmax(probs, axis=1)]
        # Confusions
        Confs[i, :, :] = confusion_matrix(truth, preds, dataset.label_values)
    # Sum all confusions
    C = np.sum(Confs, axis=0).astype(np.float32)
    # Balance with real validation proportions
    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
    # Objects IoU
    IoUs = IoU_from_confusions(C)
    # Print instance mean
    mIoU = np.mean(IoUs)
    return IoUs, mIoU


def s3dis_voting_metrics(dataset, validation_probs, val_proportions):
    Confs = []
    for i_test in range(dataset.num_validation):
        probs = validation_probs[i_test]
        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
        targets = dataset.input_labels['validation'][i_test]
        Confs += [confusion_matrix(targets, preds, dataset.label_values)]
    # Regroup confusions
    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)
    # Rescale with the right number of point per class
    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)

    return IoUs, mIoU


def s3dis_metrics(dataset, proj_probs):
    Confs = []
    for i_test in range(dataset.num_validation):
        # Get the predicted labels
        preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)
        # Confusion
        targets = dataset.validation_labels[i_test]
        Confs += [confusion_matrix(targets, preds, dataset.label_values)]

    # Regroup confusions
    C = np.sum(np.stack(Confs), axis=0)

    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)

    return IoUs, mIoU
