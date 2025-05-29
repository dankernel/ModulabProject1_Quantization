import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet18 import ResNet18 # Assuming resnet18.py is in the same directory or PYTHONPATH
import torchvision

ANCHOR_RATIOS = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes=20, num_anchors=len(ANCHOR_RATIOS), grid_size=7):
        super().__init__()
        self.backbone = ResNet18(num_classes=1000) # Pre-trained backbone
        self.grid_size = grid_size # S, feature map size (e.g., 7 for 224x224 input)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Feature extractor layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Prediction head: num_anchors * (box_coords + objectness + num_classes)
        self.pred_head = nn.Conv2d(128, self.num_anchors * (4 + 1 + self.num_classes), kernel_size=1)

    def forward(self, x):
        # Backbone: [B, 3, H, W] -> [B, 512, S, S] (e.g. S=H/32)
        x = self.backbone.forward_features(x)
        # Feature extraction: [B, 512, S, S] -> [B, 128, S, S]
        x = self.feature_extractor(x)
        # Prediction head: [B, 128, S, S] -> [B, num_anchors * (5 + C), S, S]
        x = self.pred_head(x)
        
        # Reshape for easier processing:
        # [B, num_anchors * (5 + C), S, S] -> [B, S, S, num_anchors * (5 + C)]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [B, S, S, num_anchors * (5 + C)] -> [B, S, S, num_anchors, 5 + C]
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.num_anchors, 5 + self.num_classes)
        return x


def decode_predictions(output, conf_thresh=0.25, iou_thresh=0.45, image_size=224):
    """
    Decodes raw predictions from the model output tensor.

    Args:
        output (torch.Tensor): Model output tensor of shape [B, S, S, num_anchors, 5 + C].
                               Raw logits/values for (dx, dy, dw, dh, obj, cls1, cls2...).
        conf_thresh (float): Confidence threshold for filtering detections.
                             (obj_conf * class_conf).
        iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS).
        image_size (int): The size of the input image (assumed to be square).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - boxes_nms (torch.Tensor): Detected bounding boxes after NMS, shape [N, 4] (x1, y1, x2, y2).
            - scores_nms (torch.Tensor): Corresponding confidence scores, shape [N].
            - labels_nms (torch.Tensor): Corresponding class labels, shape [N].
    """
    B, S_grid, _, num_anchors_dim, num_output_values = output.shape
    num_classes = num_output_values - 5
    
    # Ensure ANCHOR_RATIOS is a tensor for potential ops, though here used by index
    anchor_ratios_tensor = torch.tensor(ANCHOR_RATIOS, device=output.device, dtype=torch.float32)

    all_boxes_batch = []
    all_scores_batch = []
    all_labels_batch = []

    # Typically B=1 for inference, but this loop handles batch_size > 1
    for b_idx in range(B):
        batch_output_single_image = output[b_idx] # Shape: [S, S, num_anchors, 5 + C]
        
        current_image_boxes = []
        current_image_scores = []
        current_image_labels = []

        grid_cell_pixel_size = image_size / S_grid # Assuming square cells

        for i in range(S_grid):  # Grid cell row
            for j in range(S_grid):  # Grid cell column
                for a in range(num_anchors_dim): # Iterate through anchors
                    cell_prediction = batch_output_single_image[i, j, a]  # Shape: [5 + C]

                    # Extract raw values
                    dx_raw, dy_raw, dw_raw, dh_raw = cell_prediction[0:4]
                    objectness_raw = cell_prediction[4]
                    class_logits_raw = cell_prediction[5:]

                    # Apply activations
                    pred_obj_conf = torch.sigmoid(objectness_raw)
                    class_probs = torch.softmax(class_logits_raw, dim=-1) # Softmax over class logits
                    max_class_prob, pred_cls_idx = torch.max(class_probs, dim=-1)
                    
                    final_conf = pred_obj_conf * max_class_prob # Combined confidence

                    if final_conf.item() > conf_thresh:
                        # Decode bounding box coordinates
                        # Anchor dimensions are ratios relative to grid cell size
                        anchor_w_ratio, anchor_h_ratio = anchor_ratios_tensor[a]
                        
                        # Predicted center (cx, cy) relative to top-left of grid cell (i,j)
                        # and scaled by grid cell size. Sigmoid constrains offsets to (0,1).
                        cx_pixel = (j + torch.sigmoid(dx_raw).item()) * grid_cell_pixel_size
                        cy_pixel = (i + torch.sigmoid(dy_raw).item()) * grid_cell_pixel_size
                        
                        # Predicted width/height are scales for anchor base size.
                        # Anchor base size is anchor_ratio * grid_cell_size.
                        # Exp ensures positive width/height.
                        box_w_pixel = (torch.exp(dw_raw) * anchor_w_ratio).item() * grid_cell_pixel_size
                        box_h_pixel = (torch.exp(dh_raw) * anchor_h_ratio).item() * grid_cell_pixel_size

                        # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
                        x1 = cx_pixel - box_w_pixel / 2
                        y1 = cy_pixel - box_h_pixel / 2
                        x2 = cx_pixel + box_w_pixel / 2
                        y2 = cy_pixel + box_h_pixel / 2

                        # Clip coordinates to image boundaries
                        x1 = max(0.0, x1)
                        y1 = max(0.0, y1)
                        x2 = min(float(image_size), x2) # image_size is upper bound (exclusive)
                        y2 = min(float(image_size), y2)
                        
                        # Add valid boxes (positive width and height)
                        if x2 > x1 and y2 > y1:
                            current_image_boxes.append([x1, y1, x2, y2])
                            current_image_scores.append(final_conf.item())
                            current_image_labels.append(pred_cls_idx.item())
        
        if current_image_boxes:
            boxes_tensor = torch.tensor(current_image_boxes, dtype=torch.float32, device=output.device)
            scores_tensor = torch.tensor(current_image_scores, dtype=torch.float32, device=output.device)
            labels_tensor = torch.tensor(current_image_labels, dtype=torch.int64, device=output.device)

            keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_thresh)
            
            all_boxes_batch.append(boxes_tensor[keep_indices])
            all_scores_batch.append(scores_tensor[keep_indices])
            all_labels_batch.append(labels_tensor[keep_indices])

    if not all_boxes_batch: # If no boxes found in any image in the batch
         return torch.empty((0, 4), dtype=torch.float32, device=output.device), \
               torch.empty((0,), dtype=torch.float32, device=output.device), \
               torch.empty((0,), dtype=torch.int64, device=output.device)

    # For now, concatenate results if B > 1.
    # If B=1, these will just be the tensors themselves.
    final_boxes = torch.cat(all_boxes_batch, dim=0)
    final_scores = torch.cat(all_scores_batch, dim=0)
    final_labels = torch.cat(all_labels_batch, dim=0)
    
    return final_boxes, final_scores, final_labels


if __name__ == "__main__":
    num_classes_example = 20
    # num_anchors is implicitly len(ANCHOR_RATIOS) in the class default
    grid_size_example = 7 # S, typical for a 224x224 input divided by 32

    # Instantiate model
    # num_anchors will be len(ANCHOR_RATIOS) by default if not overridden.
    model = SimpleObjectDetector(num_classes=num_classes_example, grid_size=grid_size_example).eval()
    
    # Create a dummy input image tensor
    # Batch size B=1 for this example
    dummy_image = torch.randn(1, 3, 224, 224) # B, C, H, W
    
    # Get dummy model output
    with torch.no_grad(): # Important for inference
        dummy_model_output = model(dummy_image)
    
    print(f"Model output shape: {dummy_model_output.shape}")
    # Expected: [1, grid_size_example, grid_size_example, len(ANCHOR_RATIOS), 5 + num_classes_example]
    # e.g., [1, 7, 7, 3, 25]

    print(f"\n--- Decoding predictions from Model Output ---")
    # Using default conf_thresh=0.25, iou_thresh=0.45
    boxes, scores, labels = decode_predictions(dummy_model_output, image_size=224) 
    
    if boxes.numel() > 0:
        print(f"Found {boxes.shape[0]} boxes after NMS:")
        for i in range(boxes.shape[0]):
            print(f"  Box {i+1}: {boxes[i].tolist()}, Score: {scores[i].item():.4f}, Label: {labels[i].item()}")
    else:
        print("No boxes found meeting the confidence threshold from model output.")

    # Example with more controlled dummy output for specific testing
    B_test, S_test, NA_test, C_test = 1, 7, len(ANCHOR_RATIOS), num_classes_example
    # Initialize with very low confidence (logits)
    controlled_output = torch.full((B_test, S_test, S_test, NA_test, 5 + C_test), -20.0) 

    # Cell (3,3), Anchor 0: High confidence for class 2
    # dx, dy are relative to cell top-left; dw, dh are log-scales for anchor w/h
    controlled_output[0, 3, 3, 0, 0:4] = torch.tensor([0.0, 0.0, 0.0, 0.0]) # dx,dy=0 -> center of cell; dw,dh=0 -> exp(0)=1 (anchor base size)
    controlled_output[0, 3, 3, 0, 4] = 5.0  # High objectness_raw (logit) -> sigmoid near 1
    controlled_output[0, 3, 3, 0, 5+2] = 5.0 # High logit for class 2 -> softmax near 1 for this class

    # Cell (4,4), Anchor 1: Also high confidence for class 5, potentially overlapping
    controlled_output[0, 4, 4, 1, 0:4] = torch.tensor([0.0, 0.0, 0.0, 0.0]) 
    controlled_output[0, 4, 4, 1, 4] = 4.0
    controlled_output[0, 4, 4, 1, 5+5] = 4.0

    print(f"\n--- Decoding predictions from Controlled Output (conf_thresh=0.5) ---")
    boxes_c, scores_c, labels_c = decode_predictions(controlled_output, image_size=224, conf_thresh=0.5)
    
    if boxes_c.numel() > 0:
        print(f"Found {boxes_c.shape[0]} boxes from controlled output after NMS:")
        for i in range(boxes_c.shape[0]):
            # VOC_CLASSES could be used here if defined globally for printing class name
            print(f"  Box {i+1}: {boxes_c[i].tolist()}, Score: {scores_c[i].item():.4f}, Label: {labels_c[i].item()}")
    else:
        print("No boxes found from controlled output meeting the confidence threshold.")

```
