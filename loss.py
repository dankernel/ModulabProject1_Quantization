import torch
import torch.nn.functional as F

def detection_loss(pred, target):
    """
    Calculates the detection loss.
    pred: Tensor of shape [B, S, S, num_anchors, 5 + C]
          (dx, dy, dw, dh, objectness_score_logit, class_logits...)
    target: Tensor of shape [B, S, S, num_anchors, 5 + C]
            (dx, dy, dw, dh, objectness_target, one-hot_class_labels...)
    Where:
        B = batch size
        S = grid size
        num_anchors = number of anchors per grid cell
        C = number of classes (can be inferred from pred.shape[-1] - 5)
        5 = box_coords (4) + objectness_score (1)
    """

    # Extract components from predictions
    pred_boxes = pred[..., 0:4]  # dx, dy, dw, dh
    pred_obj = pred[..., 4]      # Objectness score (logit)
    pred_cls = pred[..., 5:]      # Class logits

    # Extract components from targets
    target_boxes = target[..., 0:4]  # dx, dy, dw, dh
    target_obj = target[..., 4]      # Objectness target (0.0 or 1.0)
    target_cls = target[..., 5:]      # One-hot class labels

    # --- 1. Objectness Loss ---
    # Calculated over ALL anchors (both positive and negative targets).
    # This encourages the model to learn what constitutes an object (positive)
    # and what is background (negative) for each anchor.
    # target_obj contains 0s for negative anchors and 1s for positive anchors.
    loss_obj = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='mean')

    # --- Create mask for responsible anchors ---
    # Anchors that are assigned to a ground truth object (target_obj == 1.0)
    obj_mask = target_obj == 1.0  # Shape: [B, S, S, num_anchors]

    # --- 2. Bounding Box Regression Loss ---
    # Calculated only for responsible anchors (where obj_mask is True)
    pred_boxes_responsible = pred_boxes[obj_mask]
    target_boxes_responsible = target_boxes[obj_mask]

    if pred_boxes_responsible.numel() > 0:
        # Using reduction='mean' to average loss over responsible anchors
        loss_bbox = F.smooth_l1_loss(pred_boxes_responsible, target_boxes_responsible, reduction='mean')
    else:
        # If no responsible anchors, bbox loss is 0
        loss_bbox = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # --- 3. Classification Loss ---
    # Calculated only for responsible anchors
    pred_cls_responsible = pred_cls[obj_mask]
    target_cls_responsible = target_cls[obj_mask]

    if pred_cls_responsible.numel() > 0:
        # Ensure target_cls_responsible is float for BCEWithLogitsLoss
        # Using reduction='mean' to average loss over responsible anchors
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls_responsible, target_cls_responsible.float(), reduction='mean')
    else:
        # If no responsible anchors, classification loss is 0
        loss_cls = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
    # --- Total Loss ---
    # Simple sum of components. Weights (e.g., lambda_coord, lambda_obj_pos, lambda_obj_neg) 
    # can be added here if further tuning is required.
    total_loss = loss_obj + loss_bbox + loss_cls
    
    return total_loss, loss_obj, loss_bbox, loss_cls


if __name__ == "__main__":
    B = 2  # Batch size
    S = 7  # Grid size
    num_anchors = 3  # Number of anchors
    C = 20 # Number of classes

    # Dummy predictions (logits)
    # Shape: [B, S, S, num_anchors, 5 + C]
    pred = torch.randn(B, S, S, num_anchors, 5 + C)

    # Dummy targets
    # Shape: [B, S, S, num_anchors, 5 + C]
    target = torch.zeros(B, S, S, num_anchors, 5 + C)

    # Simulate some positive anchors in the target
    # Example 1: First batch, grid cell (0,0), first anchor is responsible
    target[0, 0, 0, 0, 0:4] = torch.tensor([0.5, 0.5, 0.2, 0.2]) # dx, dy, w, h (target values)
    target[0, 0, 0, 0, 4] = 1.0  # Objectness target = 1.0 (positive anchor)
    target[0, 0, 0, 0, 5 + 3] = 1.0 # Class 3 is present (one-hot encoded)

    # Example 2: Second batch, grid cell (3,3), second anchor is responsible
    target[1, 3, 3, 1, 0:4] = torch.tensor([0.1, 0.8, 0.1, 0.1]) # dx, dy, w, h
    target[1, 3, 3, 1, 4] = 1.0  # Objectness target = 1.0
    target[1, 3, 3, 1, 5 + 10] = 1.0 # Class 10 is present

    # All other target_obj values remain 0.0 (negative anchors) by default from torch.zeros.

    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")

    total_loss, loss_obj, loss_bbox, loss_cls = detection_loss(pred, target)

    print(f"\n--- Test with some positive anchors ---")
    print(f"Total Loss: {total_loss.item()}")
    print(f"  Objectness Loss (all anchors): {loss_obj.item()}") # Calculated over all B*S*S*num_anchors
    print(f"  Bounding Box Loss (positive anchors): {loss_bbox.item()}") # Calculated over 2 positive anchors in this example
    print(f"  Classification Loss (positive anchors): {loss_cls.item()}") # Calculated over 2 positive anchors

    # Test case with no positive anchors (all target_obj are 0)
    target_no_obj = torch.zeros(B, S, S, num_anchors, 5 + C)
    total_loss_no_obj, loss_obj_no_obj, loss_bbox_no_obj, loss_cls_no_obj = detection_loss(pred, target_no_obj)
    print(f"\n--- Test with no positive anchors ---")
    print(f"Total Loss (no obj): {total_loss_no_obj.item()}") # Should be loss_obj_no_obj
    print(f"  Objectness Loss (all anchors): {loss_obj_no_obj.item()}") # Should be non-zero (all are negatives)
    print(f"  Bounding Box Loss (no positive anchors): {loss_bbox_no_obj.item()}") # Should be 0
    print(f"  Classification Loss (no positive anchors): {loss_cls_no_obj.item()}") # Should be 0
    
    # Test with all anchors positive (highly unlikely in reality, but tests logic)
    # Create random box coordinates and class labels for all anchors
    target_all_obj = torch.rand(B, S, S, num_anchors, 5 + C, device=pred.device, dtype=pred.dtype) 
    target_all_obj[..., 4] = 1.0 # Set all anchors as positive
    # Ensure class targets are valid one-hot like for BCE
    active_classes = torch.randint(0, C, (B,S,S,num_anchors), device=pred.device)
    target_all_obj[..., 5:] = F.one_hot(active_classes, num_classes=C).float()


    total_loss_all_obj, loss_obj_all_obj, loss_bbox_all_obj, loss_cls_all_obj = detection_loss(pred, target_all_obj)
    print(f"\n--- Test with all positive anchors ---")
    print(f"Total Loss (all obj): {total_loss_all_obj.item()}")
    print(f"  Objectness Loss (all obj): {loss_obj_all_obj.item()}")
    print(f"  Bounding Box Loss (all obj): {loss_bbox_all_obj.item()}")
    print(f"  Classification Loss (all obj): {loss_cls_all_obj.item()}")

```
