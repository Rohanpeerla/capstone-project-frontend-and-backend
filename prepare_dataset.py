#!/usr/bin/env python3
"""
Convert XML annotations to YOLO format and prepare dataset structure.
"""
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "car number plate dataset"
IMAGES_SOURCE = DATASET_DIR / "Indian_Number_Plates" / "Sample_Images"
ANNOTATIONS_SOURCE = DATASET_DIR / "Annotations" / "Annotations"

# Create output structure
TRAFFIC_DATASET = BASE_DIR / "traffic dataset"
TRAFFIC_IMAGES = TRAFFIC_DATASET / "images"
TRAFFIC_LABELS = TRAFFIC_DATASET / "labels"

TRAIN_IMAGES = TRAFFIC_IMAGES / "train"
TRAIN_LABELS = TRAFFIC_LABELS / "train"
VAL_IMAGES = TRAFFIC_IMAGES / "val"
VAL_LABELS = TRAFFIC_LABELS / "val"

def xml_to_yolo(xml_path, img_width, img_height):
    """Convert Pascal VOC XML to YOLO format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        # Only include number plate annotations
        if name.lower() != "number plate":
            continue
            
        bndbox = obj.find('bndbox')
        
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        
        # Convert to YOLO format (center x, center y, width, height) normalized
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Class 0 = number plate
        class_id = 0
        
        objects.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return objects

def prepare_dataset():
    """Prepare dataset for YOLO training."""
    print("Preparing dataset...")
    
    # Create directories
    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = sorted([f for f in IMAGES_SOURCE.glob("*.jpg")])
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("ERROR: No images found!")
        return False
    
    # Split into train (80%) and val (20%)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Process training images
    for img_path in train_files:
        img_name = img_path.stem
        xml_path = ANNOTATIONS_SOURCE / f"{img_name}.xml"
        
        if not xml_path.exists():
            print(f"WARNING: No annotation for {img_name}")
            continue
        
        # Read image to get dimensions
        try:
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"ERROR: Could not read {img_path}")
                continue
            h, w = img.shape[:2]
        except:
            print(f"ERROR: Could not read image dimensions for {img_path}")
            continue
        
        # Convert annotations
        objects = xml_to_yolo(str(xml_path), w, h)
        
        # Copy image
        shutil.copy(str(img_path), str(TRAIN_IMAGES / img_path.name))
        
        # Write YOLO annotations
        label_path = TRAIN_LABELS / f"{img_name}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(objects))
    
    # Process validation images
    for img_path in val_files:
        img_name = img_path.stem
        xml_path = ANNOTATIONS_SOURCE / f"{img_name}.xml"
        
        if not xml_path.exists():
            print(f"WARNING: No annotation for {img_name}")
            continue
        
        # Read image to get dimensions
        try:
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"ERROR: Could not read {img_path}")
                continue
            h, w = img.shape[:2]
        except:
            print(f"ERROR: Could not read image dimensions for {img_path}")
            continue
        
        # Convert annotations
        objects = xml_to_yolo(str(xml_path), w, h)
        
        # Copy image
        shutil.copy(str(img_path), str(VAL_IMAGES / img_path.name))
        
        # Write YOLO annotations
        label_path = VAL_LABELS / f"{img_name}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(objects))
    
    print("✓ Dataset prepared successfully!")
    return True

if __name__ == "__main__":
    success = prepare_dataset()
    if success:
        print("\nNow run: python train_traffic_model.py")
