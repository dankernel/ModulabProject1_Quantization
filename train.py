import os # For path checking and dummy data creation
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# Assuming these modules are in the same directory or accessible via PYTHONPATH
from voc import VOCDataset, VOC_CLASSES # VOC_CLASSES is used for num_classes
from detector import SimpleObjectDetector # ANCHOR_RATIOS is defined in detector.py
from loss import detection_loss

# Define constants for clarity and easy modification
IMAGE_SIZE = 224
# GRID_SIZE is determined by the model architecture (e.g., ResNet18 backbone typically H/32)
# For a 224x224 input, S = 224 / 32 = 7
GRID_SIZE = 7 
# NUM_ANCHORS is derived from len(ANCHOR_RATIOS) in detector.py,
# SimpleObjectDetector defaults to this.

def train(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    # Updated VOCDataset instantiation:
    # - Pass image_size (required by the modified VOCDataset for normalization)
    # - Pass grid_size (used by VOCDataset for target encoding, should match model's grid_size)
    dataset = VOCDataset(
        root="VOCdevkit/VOC2007", # Adjust this path if your dataset is elsewhere
        image_set="trainval",
        transform=transform, 
        image_size=IMAGE_SIZE, 
        grid_size=GRID_SIZE
    )

    loader = DataLoader(
        dataset, 
        batch_size=16, # Adjusted batch size from 128, as 128 might be too large for many systems
        shuffle=True,
        num_workers=2, # Common practice for faster data loading if CPU allows
        pin_memory=True if device == "cuda" else False # Useful for CUDA to speed up CPU to GPU transfer
    )

    # Model instantiation:
    # - num_classes is derived from VOC_CLASSES.
    # - num_anchors defaults to len(ANCHOR_RATIOS) in SimpleObjectDetector.
    # - grid_size in SimpleObjectDetector is the expected feature map size.
    model = SimpleObjectDetector(
        num_classes=len(VOC_CLASSES),
        grid_size=GRID_SIZE 
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Adjusted learning rate from 3e-4

    num_epochs = 20 # Number of epochs for training
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_cls_loss = 0.0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in progress_bar:
            images = images.to(device)
            # Targets from VOCDataset are already shaped [S, S, num_anchors, 5+C] per item.
            # DataLoader batches them to [B, S, S, num_anchors, 5+C].
            targets = targets.to(device)

            # Forward pass
            preds = model(images) # Output shape: [B, S, S, num_anchors, 5+C]
            
            # Calculate loss
            # detection_loss now returns (total_loss, loss_obj, loss_bbox, loss_cls)
            total_loss_val, loss_obj_val, loss_bbox_val, loss_cls_val = detection_loss(preds, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss_val.backward()
            optimizer.step()
            
            # Accumulate losses for epoch summary
            epoch_total_loss += total_loss_val.item()
            epoch_obj_loss += loss_obj_val.item()
            epoch_bbox_loss += loss_bbox_val.item()
            epoch_cls_loss += loss_cls_val.item()
            
            # Update progress bar postfix with current batch losses
            progress_bar.set_postfix({
                "Total": f"{total_loss_val.item():.4f}",
                "Obj": f"{loss_obj_val.item():.4f}",
                "BBox": f"{loss_bbox_val.item():.4f}",
                "Class": f"{loss_cls_val.item():.4f}"
            })

        # Calculate average losses for the epoch
        avg_epoch_total_loss = epoch_total_loss / len(loader)
        avg_epoch_obj_loss = epoch_obj_loss / len(loader)
        avg_epoch_bbox_loss = epoch_bbox_loss / len(loader)
        avg_epoch_cls_loss = epoch_cls_loss / len(loader)

        print(f"Epoch {epoch+1}/{num_epochs} Summary: "
              f"Avg Total Loss = {avg_epoch_total_loss:.4f}, "
              f"Avg Obj Loss = {avg_epoch_obj_loss:.4f}, "
              f"Avg BBox Loss = {avg_epoch_bbox_loss:.4f}, "
              f"Avg Class Loss = {avg_epoch_cls_loss:.4f}")

        # Save model checkpoint periodically
        # Original saved as model_epoch_{epoch}.pth, changed to epoch+1 for 1-based epoch num
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs: # Save every 5 epochs and at the end
            checkpoint_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

    final_model_path = "model_final.pth" # Original was model.pth
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved as {final_model_path}")

def create_dummy_voc_structure(voc_root):
    """Creates a minimal dummy VOC dataset structure for testing purposes if Pillow is installed."""
    print(f"Attempting to create dummy VOC structure at {voc_root} for testing.")
    try:
        from PIL import Image # Pillow is required: pip install Pillow
        
        # Create directories
        os.makedirs(os.path.join(voc_root, "JPEGImages"), exist_ok=True)
        os.makedirs(os.path.join(voc_root, "Annotations"), exist_ok=True)
        os.makedirs(os.path.join(voc_root, "ImageSets", "Main"), exist_ok=True)

        # Create a dummy image file
        dummy_img_name = "dummy_000001.jpg"
        dummy_img_path = os.path.join(voc_root, "JPEGImages", dummy_img_name)
        if not os.path.exists(dummy_img_path):
            dummy_img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='red') # Use IMAGE_SIZE
            dummy_img.save(dummy_img_path)

        # Create a dummy annotation file
        dummy_xml_name = "dummy_000001.xml"
        dummy_xml_path = os.path.join(voc_root, "Annotations", dummy_xml_name)
        if not os.path.exists(dummy_xml_path):
            # Ensure the class name is in VOC_CLASSES for consistency
            obj_class_name = "cat" if "cat" in VOC_CLASSES else VOC_CLASSES[0] if VOC_CLASSES else "object"
            dummy_xml_content = f"""<annotation>
                <folder>VOC2007</folder>
                <filename>{dummy_img_name}</filename>
                <size><width>{IMAGE_SIZE}</width><height>{IMAGE_SIZE}</height><depth>3</depth></size>
                <object>
                    <name>{obj_class_name}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
                    <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>{IMAGE_SIZE-50}</xmax><ymax>{IMAGE_SIZE-50}</ymax></bndbox>
                </object>
            </annotation>"""
            with open(dummy_xml_path, "w") as xml_f:
                xml_f.write(dummy_xml_content)
        
        # Create trainval.txt if it doesn't exist or is empty
        trainval_path = os.path.join(voc_root, "ImageSets", "Main", "trainval.txt")
        if not os.path.exists(trainval_path) or os.path.getsize(trainval_path) == 0:
             with open(trainval_path, "w") as f:
                f.write("dummy_000001\n") # Add dummy file ID

        print("Dummy VOC structure created/verified successfully.")
        return True
    except ImportError:
        print("Pillow (PIL) is not installed. Cannot create dummy image. Please run: pip install Pillow")
        return False
    except Exception as e:
        print(f"Could not create/verify dummy VOC structure: {e}")
        return False

if __name__ == "__main__":
    voc_data_path = "VOCdevkit/VOC2007" # Standard path for VOC data
    
    # Check if the essential dataset directories and trainval.txt exist
    jpeg_dir = os.path.join(voc_data_path, "JPEGImages")
    ann_dir = os.path.join(voc_data_path, "Annotations")
    imgsets_main_dir = os.path.join(voc_data_path, "ImageSets", "Main")
    trainval_file = os.path.join(imgsets_main_dir, "trainval.txt")

    # Check if dataset seems valid or if trainval.txt is empty
    dataset_valid = (os.path.exists(jpeg_dir) and os.path.isdir(jpeg_dir) and \
                     os.path.exists(ann_dir) and os.path.isdir(ann_dir) and \
                     os.path.exists(trainval_file) and os.path.isfile(trainval_file) and \
                     os.path.getsize(trainval_file) > 0)

    if not dataset_valid:
        print(f"VOC dataset not found, incomplete, or trainval.txt is empty at '{voc_data_path}'.")
        if not create_dummy_voc_structure(voc_data_path):
            print("Failed to create dummy dataset. Please ensure VOC dataset is correctly set up or install Pillow.")
            exit(1) # Exit if dummy creation also fails or is not possible
        else:
            print("Proceeding with dummy dataset for testing purposes.")
    else:
        print(f"Found valid VOC dataset at '{voc_data_path}'.")

    train()
```
