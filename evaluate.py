import torch
import torchvision
from torch.utils.data import DataLoader
from voc import VOCDataset, VOC_CLASSES
from detector import SimpleObjectDetector
from detector import decode_predictions
from tqdm import tqdm
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    iou = torchvision.ops.box_iou(box1, box2)  # [N, M]
    return iou


def evaluate(model, dataloader, device="cpu", iou_thresh=0.5, conf_thresh=0.5):
    model.eval()
    ap_per_class = defaultdict(list)  # {class_id: list of APs}

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            targets = [t for t in targets]  # list of [S, S, 4+C] tensors

            preds = model(images)  # [B, S, S, 4+C]
            for pred, gt in zip(preds, targets):
                pred_boxes, pred_scores, pred_labels = decode_predictions(
                    pred.unsqueeze(0), conf_thresh=conf_thresh, iou_thresh=iou_thresh
                )

                # GT 복원: VOC-style target [S, S, 4+C]
                gt = gt.cpu()
                gt_boxes, gt_labels = [], []
                for i in range(gt.shape[0]):
                    for j in range(gt.shape[1]):
                        if gt[i, j, 4:].sum() > 0:
                            label = torch.argmax(gt[i, j, 4:])
                            box = gt[i, j, :4]
                            cx, cy, w, h = box.tolist()
                            cx = (j + cx) / 7
                            cy = (i + cy) / 7
                            x1 = (cx - w / 2) * 224
                            y1 = (cy - h / 2) * 224
                            x2 = (cx + w / 2) * 224
                            y2 = (cy + h / 2) * 224
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_labels.append(label.item())

                if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                    continue

                gt_boxes = torch.tensor(gt_boxes)
                gt_labels = torch.tensor(gt_labels)

                ious = compute_iou(pred_boxes, gt_boxes)

                for pred_idx, pred_label in enumerate(pred_labels):
                    max_iou, gt_idx = ious[pred_idx].max(0)
                    if max_iou > iou_thresh and pred_label == gt_labels[gt_idx]:
                        ap_per_class[pred_label.item()].append(1)  # TP
                    else:
                        ap_per_class[pred_label.item()].append(0)  # FP

    # mAP 계산
    aps = []
    for cls in range(len(VOC_CLASSES)):
        if cls in ap_per_class:
            ap = np.mean(ap_per_class[cls])
            print(f"Class {VOC_CLASSES[cls]}: AP = {ap:.4f}")
            aps.append(ap)
        else:
            aps.append(0.0)

    print(f"mAP: {np.mean(aps):.4f}")


if __name__ == "__main__":
    device = "cuda"
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    dataset = VOCDataset("VOCdevkit/VOC2007", transform=transform, image_set="val")

    def collate_fn(batch):
        images, targets = zip(*batch)
        return torch.stack(images), torch.stack(targets)

    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

    model = SimpleObjectDetector(num_classes=len(VOC_CLASSES))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)

    evaluate(model, dataloader, device=device)
