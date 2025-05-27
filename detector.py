import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet18 import ResNet18
import torchvision


class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes=20, grid_size=13):
        super().__init__()
        self.backbone = ResNet18(num_classes=1000)  # fully-connected는 사용 안 함
        self.grid_size = grid_size
        self.num_classes = num_classes

        # Feature 추출용 conv block
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Detection Head
        self.pred_head = nn.Conv2d(128, (4 + num_classes), kernel_size=1)

    def forward(self, x):
        # [B, 3, 224, 224] -> backbone -> [B, 512, 7, 7]
        x = self.backbone.forward_features(x)  # 새로 정의 (아래 참고)

        # Feature -> Head
        x = self.feature_extractor(x)  # [B, 128, 7, 7]
        x = self.pred_head(x)  # [B, 4+C, 7, 7]

        # [B, (4+C), 7, 7] -> [B, 7, 7, 4+C]
        x = x.permute(0, 2, 3, 1).contiguous()

        return x


def decode_predictions(
    output, conf_thresh=0.5, iou_thresh=0.5, num_classes=20, image_size=224
):
    """
    output: Tensor of shape [1, 7, 7, 24] = [B, S, S, 4 + num_classes]
    returns: boxes [N, 4], scores [N], labels [N]
    """
    S = output.shape[1]
    output = output[0]  # [7, 7, 24]
    boxes = []
    scores = []
    labels = []

    for i in range(S):
        for j in range(S):
            cell = output[i, j]  # shape: [4 + num_classes]
            cx, cy, w, h = cell[:4]
            class_logits = cell[4:]
            class_probs = F.softmax(class_logits, dim=0)

            conf, cls = torch.max(class_probs, dim=-1)

            if conf > conf_thresh and w > 0.01 and h > 0.01:
                # grid cell (i,j) 기준 위치 → 이미지 상대 좌표로 변환
                cx = (j + cx.item()) / S
                cy = (i + cy.item()) / S
                w = w.item()
                h = h.item()

                # (cx, cy, w, h) → (x1, y1, x2, y2)
                x1 = max(cx - w / 2, 0)
                y1 = max(cy - h / 2, 0)
                x2 = min(cx + w / 2, 1)
                y2 = min(cy + h / 2, 1)

                # pixel 단위로 스케일링 (원본 이미지 기준)
                x1 *= image_size
                y1 *= image_size
                x2 *= image_size
                y2 *= image_size

                boxes.append([x1, y1, x2, y2])
                scores.append(conf.item())
                labels.append(cls.item())

    if len(boxes) == 0:
        return [], [], []

    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    labels = torch.tensor(labels)

    # NMS 적용
    keep = torchvision.ops.nms(boxes, scores, iou_thresh)

    return boxes[keep], scores[keep], labels[keep]


if __name__ == "__main__":
    model = SimpleObjectDetector(num_classes=20).eval()
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    # torch.Size([1, 7, 7, 24]), out[0, i, j] = [cx, cy, w, h, class_logits...]
    print("output shape:", out.shape)

    boxes, scores, labels = decode_predictions(out)
    print("boxes:", boxes)
    print("scores:", scores)
    print("labels:", labels)
