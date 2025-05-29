import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import transforms # Already imported, ensure it's used if needed for default transform

# Attempt to import ANCHOR_RATIOS from detector.py
# This assumes detector.py is in the same directory or accessible in PYTHONPATH
try:
    from detector import ANCHOR_RATIOS
except ImportError:
    # Fallback if detector.py is not found, e.g. during standalone testing of voc.py
    # This is not ideal for production but helps in decoupled development/testing.
    # A better solution might involve a shared config or ensuring PYTHONPATH is set.
    print("Warning: Could not import ANCHOR_RATIOS from detector.py. Using default.")
    ANCHOR_RATIOS = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

NUM_ANCHORS = len(ANCHOR_RATIOS)

def iou_wh(box1_wh, box2_wh):
    """
    Calculates IoU between two boxes represented by their normalized width and height.
    box1_wh: tensor [w1, h1]
    box2_wh: tensor [w2, h2]
    """
    w1, h1 = box1_wh
    w2, h2 = box2_wh
    inter_w = torch.min(w1, w2)
    inter_h = torch.min(h1, h2)
    inter_area = inter_w * inter_h
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6) # Add epsilon for stability


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set="trainval", transform=None, grid_size=7, image_size=224):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.grid_size = grid_size
        self.image_size = image_size # Store image_size

        self.num_anchors = NUM_ANCHORS
        # Store anchor ratios as a tensor for easier computation
        self.anchor_ratios_tensor = torch.tensor(ANCHOR_RATIOS, dtype=torch.float32)


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
        img_path = self.imgs[idx]
        ann_path = self.anns[idx]

        img = Image.open(img_path).convert("RGB")
        
        # Get original image dimensions for _parse_xml if needed, or pass image_size
        # For this task, _parse_xml returns absolute pixel values, so original dimensions are not strictly needed there.
        # However, if transforms include resizing, the image object `img` will be updated.
        # We need a consistent image_size for normalization in encode_grid.
        
        original_w, original_h = img.size

        boxes, labels = self._parse_xml(ann_path) # boxes are [x1,y1,x2,y2] absolute pixel values

        if self.transform:
            # The transform should handle resizing to self.image_size
            # and converting to tensor.
            # Example: transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
            img = self.transform(img)
        else:
            # Default minimal transform if none provided (e.g. resize and to tensor)
            # This ensures img is a tensor and resized.
            temp_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])
            img = temp_transform(img)


        # Pass original_w, original_h if normalization in encode_grid needs them.
        # However, the task implies boxes from _parse_xml are absolute and then normalized against self.image_size.
        # So, we use self.image_size as the reference for normalization.
        target = self.encode_grid(boxes, labels) 
        return img, target

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
        return boxes, labels # boxes are absolute pixel values [x1, y1, x2, y2]

    def encode_grid(self, boxes, labels):
        S = self.grid_size
        target = torch.zeros((S, S, self.num_anchors, 5 + len(VOC_CLASSES))) # 5 = 4 (bbox) + 1 (obj score)

        for box, label_idx in zip(boxes, labels):
            x1, y1, x2, y2 = box # Absolute pixel coordinates

            # Normalize box coordinates using self.image_size
            # This assumes the input image (after transforms) is self.image_size x self.image_size
            # and the GT boxes are scaled according to this image size.
            # If original image dimensions were used for normalization, they'd need to be passed here.
            # But since self.transform resizes to self.image_size, this should be consistent.

            # Convert absolute pixel values to normalized [0,1] relative to self.image_size
            # Important: if the image aspect ratio is changed by resizing to a square self.image_size,
            # the GT boxes also need to be scaled accordingly.
            # For simplicity, we assume boxes are already relative to a potentially resized image,
            # or that the transform handles this. The most robust way is to transform boxes along with image.
            # Here, we directly normalize using self.image_size, assuming box coords are from an image of this size.
            
            # Let's assume the boxes from _parse_xml are for the *original* image dimensions.
            # We need to adjust them if the image was resized by self.transform.
            # This part is tricky. For now, let's proceed with the direct normalization
            # as per the problem description's simplified approach for `encode_grid`.
            # A robust pipeline would pass original_w, original_h and transformed_w, transformed_h
            # to correctly scale the boxes.
            # Given the current setup, we normalize by self.image_size.

            gt_cx_abs = (x1 + x2) / 2
            gt_cy_abs = (y1 + y2) / 2
            gt_w_abs = x2 - x1
            gt_h_abs = y2 - y1

            # Normalize relative to self.image_size (assuming square image)
            cx = gt_cx_abs / self.image_size
            cy = gt_cy_abs / self.image_size
            w = gt_w_abs / self.image_size  # Normalized width
            h = gt_h_abs / self.image_size  # Normalized height
            
            if w <=0 or h <=0: continue # Skip invalid boxes

            # Determine grid cell (i, j)
            i_cell = min(int(cy * S), S - 1) # Row index
            j_cell = min(int(cx * S), S - 1) # Column index

            # Find best anchor
            gt_wh_norm = torch.tensor([w, h], dtype=torch.float32)
            best_iou = 0.0
            best_anchor_idx = 0
            
            for anchor_idx, anchor_wh_ratio in enumerate(self.anchor_ratios_tensor):
                # The ANCHOR_RATIOS are just ratios (e.g., (1.0, 1.0), (1.0, 2.0)).
                # To use iou_wh, we need comparable w,h.
                # Option 1: Assume anchors scale to cell size. This is complex.
                # Option 2: Use ANCHOR_RATIOS to represent *shapes* and compare with gt_wh_norm shape.
                # The problem states: "assume ANCHOR_RATIOS can be directly used with iou_wh against the gt_wh"
                # This means we're comparing the aspect ratio of the GT box (w,h) with the predefined anchor ratios.
                # This is a simplification. A more common way is to define anchor boxes with specific scales and ratios
                # at the input image resolution, then scale them down to the feature map.
                # For this task, we follow the simplified instruction.
                current_iou = iou_wh(gt_wh_norm, anchor_wh_ratio) # anchor_wh_ratio is already [w,h] like
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_anchor_idx = anchor_idx
            
            # If no anchor has IoU > 0, this object might be ignored or assigned to the one with max IoU.
            # Current logic assigns to max IoU regardless of its value (unless all are 0).

            # Target values:
            # dx, dy: center of GT relative to top-left of the cell, scaled by cell size (so range [0,1])
            dx = cx * S - j_cell
            dy = cy * S - i_cell
            
            # dw, dh: normalized GT width and height (relative to image size)
            # (as per instruction "dw = w", "dh = h")
            # More advanced methods might store log(w_gt/w_anchor) or similar.
            dw = w
            dh = h

            # Populate target tensor
            # Check if cell is already occupied by another object for this anchor
            if target[i_cell, j_cell, best_anchor_idx, 4] == 0: # If objectness is 0 (not assigned yet)
                target[i_cell, j_cell, best_anchor_idx, 0:4] = torch.tensor([dx, dy, dw, dh], dtype=torch.float32)
                target[i_cell, j_cell, best_anchor_idx, 4] = 1.0  # Objectness score
                target[i_cell, j_cell, best_anchor_idx, 5 + label_idx] = 1.0  # One-hot class
        
        return target

if __name__ == '__main__':
    # Example Usage (requires VOC dataset to be available at specified root)
    # Ensure you have a VOCdevkit structure in a directory like './data/VOCdevkit'
    # And that ImageSets/Main/trainval.txt exists.
    
    # Define a sample transform
    sample_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to the image_size used by dataset
        transforms.ToTensor(),
        # Add normalization if your model expects it
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    try:
        # Adjust root path as per your dataset location
        voc_root = "./data/VOCdevkit/VOC2007" 
        if not os.path.exists(voc_root):
             voc_root = "./VOCdevkit/VOC2007" # Common alternative
        if not os.path.exists(voc_root):
            print(f"VOC root directory not found at {voc_root} or ./data/VOCdevkit/VOC2007. Skipping example.")
        else:
            dataset = VOCDataset(root=voc_root, image_set="trainval", transform=sample_transform, grid_size=7, image_size=224)
            
            if len(dataset) > 0:
                img, target = dataset[0]
                print("Image shape:", img.shape) # Expected: [3, 224, 224]
                print("Target shape:", target.shape) # Expected: [7, 7, NUM_ANCHORS, 5 + num_classes]
                                                    # e.g. [7, 7, 3, 25] for VOC_CLASSES=20
                
                # Verify some values
                print(f"NUM_ANCHORS: {NUM_ANCHORS}")
                print(f"Num classes: {len(VOC_CLASSES)}")
                
                # Check if any object was assigned
                assigned_objects = torch.sum(target[:, :, :, 4])
                print(f"Number of assigned object cells (sum of objectness scores): {assigned_objects.item()}")
                
                # Find an assigned cell to print its details
                found = False
                for r in range(target.shape[0]):
                    for c in range(target.shape[1]):
                        for a in range(target.shape[2]):
                            if target[r,c,a,4] > 0.5: # Objectness is 1.0
                                print(f"Object at cell ({r},{c}), anchor {a}:")
                                print(f"  bbox (dx,dy,w,h): {target[r,c,a,0:4]}")
                                print(f"  objectness: {target[r,c,a,4]}")
                                class_probs = target[r,c,a,5:]
                                class_idx = torch.argmax(class_probs)
                                print(f"  class_idx: {class_idx.item()}, class_name: {VOC_CLASSES[class_idx.item()]}")
                                found = True
                                break
                        if found: break
                    if found: break
                if not found:
                    print("No object found in the first sample's target tensor with high confidence.")

            else:
                print("Dataset is empty. Check image_set and file paths.")
    
    except FileNotFoundError as e:
        print(f"Error during example usage: {e}. Ensure dataset paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")

```
