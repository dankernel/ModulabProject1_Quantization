import torch
import torch.nn.functional as F


def detection_loss(pred, target, num_classes=20):
    """
    pred, target: [B, S, S, 4+C]
    """
    B, S, _, _ = pred.shape

    # 객체가 있는 셀 mask
    obj_mask = target[..., 4:].sum(-1) > 0  # [B, S, S]

    # bbox regression loss: 객체가 있는 셀만 계산
    bbox_pred = pred[..., :4][obj_mask]
    bbox_true = target[..., :4][obj_mask]
    if bbox_pred.numel() == 0:
        loss_bbox = 0.0
    else:
        loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_true)

    # classification loss: 전체 셀
    cls_pred = pred[..., 4:]
    cls_true = target[..., 4:]
    # 이진 분류(one-hot), 다수 클래스 있음
    loss_cls = F.binary_cross_entropy_with_logits(cls_pred, cls_true)

    return loss_bbox + loss_cls


if __name__ == "__main__":
    pred = torch.randn(8, 7, 7, 24)
    target = torch.randn(8, 7, 7, 24)
    loss = detection_loss(pred, target)
    print(loss)
