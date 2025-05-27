import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from voc import VOCDataset, VOC_CLASSES
from detector import SimpleObjectDetector
from detector import decode_predictions  # 예측 decode 함수
import random


def show_detections(
    img_tensor, boxes, scores, labels, class_names, score_threshold=0.1
):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # [C, H, W] → [H, W, C]

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue
        x1, y1, x2, y2 = boxes[i]
        label = class_names[labels[i]]
        score = scores[i]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 5,
            f"{label}: {score:.2f}",
            color="white",
            bbox=dict(facecolor="red", alpha=0.5, edgecolor="none"),
            fontsize=8,
        )

    plt.axis("off")
    plt.show()


def visualize_prediction(model_path="model.pth", device="cpu", index=None):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    dataset = VOCDataset("VOCdevkit/VOC2007", transform=transform, image_set="val")

    if index is None:
        index = random.randint(0, len(dataset) - 1)

    img, target = dataset[index]
    img = img.to(device)

    model = SimpleObjectDetector(num_classes=len(VOC_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(img.unsqueeze(0))  # [1, 7, 7, 24]
        boxes, scores, labels = decode_predictions(
            output, conf_thresh=0.5, iou_thresh=0.5
        )

    print(f"Visualizing image index {index} with {len(boxes)} predicted boxes")
    show_detections(img, boxes, scores, labels, VOC_CLASSES)


if __name__ == "__main__":
    visualize_prediction(device="cpu", model_path="model.pth")
