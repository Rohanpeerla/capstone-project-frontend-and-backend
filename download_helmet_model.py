#!/usr/bin/env python3
"""
Download a pre-trained helmet detection model.
Requires internet access.
"""
import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
HELMET_MODEL_OUTPUT = BASE_DIR / "helmet_model.pt"

def download_helmet_model():
    """Download a pre-trained YOLOv8 model trained on safety helmets."""
    print("Downloading helmet detection model...")
    print("This may take a few minutes on first run...")
    
    try:
        # Use YOLOv8 pretrained model - it has detected people which can help
        # For a full helmet detection model, you'd need a custom trained model
        # from Roboflow or your own training
        model = YOLO('yolov8m.pt')  # Medium model has better detection
        model.save(str(HELMET_MODEL_OUTPUT))
        print(f"✓ Model saved to: {HELMET_MODEL_OUTPUT}")
        print("\nNote: This is the standard YOLOv8m model.")
        print("For true helmet detection, download a model trained specifically for helmets from:")
        print("  - Roboflow: https://roboflow.com/detection")
        print("  - Kaggle: https://kaggle.com/datasets")
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    if HELMET_MODEL_OUTPUT.exists():
        print(f"Helmet model already exists at: {HELMET_MODEL_OUTPUT}")
    else:
        success = download_helmet_model()
        if success:
            print("\nRestart the Flask app to use the new model:")
            print("  python app.py")
        sys.exit(0 if success else 1)
