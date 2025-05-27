import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import transforms

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set="trainval", transform=None, grid_size=7):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.grid_size = grid_size

        img_dir = os.path.join(root, "JPEGImages")
        ann_dir = os.path.join(root, "Annotations")
        split_file = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")

        with open(split_file) as f:
            self.ids = [line.strip() for line in f]

        self.imgs = [os.path.join(img_dir, f"{id_}.jpg") for id_ in self.ids]
        self.anns = [os.path.join(ann_dir, f"{id_}.xml") for id_ in self.ids]

        self.class2idx = {name: i for i, name in enumerate(VOC_CLASSES)}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        boxes, labels = self._parse_xml(self.anns[idx])

        if self.transform:
            img = self.transform(img)

        target = self.encode_grid(boxes, labels)
        return img, target  # target shape: [S, S, 4+C]

    def _parse_xml(self, ann_path):
        root = ET.parse(ann_path).getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            label = obj.find("name").text.lower().strip()
            if label not in self.class2idx:
                continue
            cls_idx = self.class2idx[label]

            bnd = obj.find("bndbox")
            x1 = float(bnd.find("xmin").text)
            y1 = float(bnd.find("ymin").text)
            x2 = float(bnd.find("xmax").text)
            y2 = float(bnd.find("ymax").text)
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_idx)
        return boxes, labels

    def encode_grid(self, boxes, labels, image_size=224):
        S = self.grid_size
        target = torch.zeros((S, S, 4 + len(VOC_CLASSES)))  # 4+bbox + C

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            # normalize
            cx = (x1 + x2) / 2 / image_size
            cy = (y1 + y2) / 2 / image_size
            w = (x2 - x1) / image_size
            h = (y2 - y1) / image_size

            i = min(int(cy * S), S - 1)
            j = min(int(cx * S), S - 1)

            target[i, j, 0:4] = torch.tensor([cx * S - j, cy * S - i, w, h])
            target[i, j, 4 + label] = 1.0  # one-hot class
        return target
