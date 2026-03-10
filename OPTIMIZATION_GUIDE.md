# Helmet Detection System - Optimization & Usage Guide

## Overview
This document explains how to optimize your helmet detection system for fast responses and accurate number plate character detection.

---

## Features Implemented

### 1. **Helmet Detection**
- **Output Format**: "Helmet Detected: Yes" or "Helmet Detected: No"
- **Detection Count**: Shows number of helmets detected when helmets are present
- **Supported Endpoints**:
  - `/detect_image` - Single image analysis
  - `/detect_video` - Video file analysis  
  - `/webcam` - Real-time webcam stream

### 2. **Number Plate Character Detection**
- **Technology**: EasyOCR with multi-variant processing
- **Features**:
  - Automatic text cleaning and validation
  - Indian license plate format support
  - Character-to-digit mapping (O→0, I→1, etc.)
  - Confidence scoring
- **Fast Mode**: Enabled for image uploads (uses 1 OCR variant for speed)
- **Standard Mode**: Used for video/webcam (uses multiple variants for accuracy)

### 3. **Performance Optimizations**

#### Response Time Improvements:
1. **Image Inference Scaling**
   - Automatically resizes large images to fit MAX_INFERENCE_DIM
   - Default: 640px (faster than 1080p)
   - Reduces memory usage and inference time

2. **OCR Optimization**
   - **Fast Mode** (Images): 1 OCR variant = ~50% faster
   - **Standard Mode** (Video): 3 OCR variants = better accuracy
   - Adaptive processing based on input type

3. **Frame Sampling**
   - Video: Processes every 8th frame for OCR
   - Reduces computation by ~87.5%
   - Maintains real-time display quality

4. **Result Caching**
   - Image results cached using SHA-256 file hash
   - Default: 32 images cached
   - Instant response for duplicate submissions

5. **Lazy Loading**
   - Models load only when needed
   - Runtime preloads models on startup

---

## Configuration for Speed vs Accuracy

### **Fast Response (Best for Real-time)**
```
FRAME_SAMPLE_INTERVAL=8              # Every 8th frame
MAX_INFERENCE_DIM=480                # Smaller = faster
YOLO_IMAGE_SIZE=480
OCR_VARIANT_LIMIT=1                  # Single variant
MAX_VEHICLE_CROPS=1                  # One vehicle max
```
**Expected Response Time**: 100-300ms per image

### **Balanced (Default)**
```
FRAME_SAMPLE_INTERVAL=8
MAX_INFERENCE_DIM=640                # Default
YOLO_IMAGE_SIZE=480
OCR_VARIANT_LIMIT=1
MAX_VEHICLE_CROPS=1
```
**Expected Response Time**: 150-400ms per image

### **High Accuracy (Best for Batch Processing)**
```
FRAME_SAMPLE_INTERVAL=4              # Every 4th frame
MAX_INFERENCE_DIM=1280               # Larger = accurate
YOLO_IMAGE_SIZE=640
OCR_VARIANT_LIMIT=3                  # Multiple variants
MAX_VEHICLE_CROPS=3                  # Multiple vehicles
```
**Expected Response Time**: 500-1000ms per image

---

## API Response Format

### Image Detection Response
```json
{
  "image_path": "/static/results/result_image.jpg",
  "helmet_status": "Yes",
  "helmet_detected": true,
  "helmet_count": 1,
  "detection_summary": "Helmet Detected: Yes (1 detected)",
  "plate_numbers": ["KA01AB1234"],
  "helmet_detection_supported": true
}
```

### Video Detection Response
```json
{
  "video_path": "/static/results/video_1234567890.mp4",
  "helmet_status": "No",
  "helmet_detected": false,
  "helmet_count": 0,
  "detection_summary": "Helmet Detected: No",
  "plate_numbers": ["MH02CD5678", "DL03EF9012"],
  "helmet_detection_supported": true
}
```

---

## Model Requirements

### Required Models
1. **Helmet Model** (helmet_model.pt)
   - Detects helmet presence in images
   - Alternative: TRAFFIC_MODEL_PATH if helmet model not found

2. **Plate Model** (plate_model.pt)
   - Detects number plate bounding boxes
   - Optional: Falls back to vehicle detection if unavailable

3. **Traffic Model** (traffic_model.pt)
   - Backup for helmet and plate detection

### Default Models
- YOLOv8n.pt is used if specific models aren't found
- Ensure models are in the project root directory

---

## Performance Tuning Guide

### To Speed Up Responses:
1. Reduce `MAX_INFERENCE_DIM` → 480 (from 640)
2. Set `OCR_VARIANT_LIMIT=1`
3. Reduce `YOLO_IMAGE_SIZE` → 416
4. Increase `FRAME_SAMPLE_INTERVAL` → 16
5. Reduce `IMAGE_CACHE_SIZE` if memory limited

### To Improve Accuracy:
1. Increase `MAX_INFERENCE_DIM` → 1280
2. Set `OCR_VARIANT_LIMIT=3`
3. Increase `YOLO_IMAGE_SIZE` → 640
4. Reduce `FRAME_SAMPLE_INTERVAL` → 4
5. Increase `MAX_VEHICLE_CROPS` → 3

### Memory Management:
- Image Cache Size: Controls RAM usage (32 default)
- Inference Dimension: Larger = more memory
- GPU Memory: Use if available (ultralytics auto-detects)

---

## Troubleshooting

### Slow Image Response (>500ms)
- Check if IMAGE_CACHE_SIZE is saturated
- Reduce MAX_INFERENCE_DIM
- Ensure GPU is available (if configured)

### Missed Helmet Detections
- Verify HELMET_MODEL_PATH points to correct model
- Check image quality and lighting
- Helmets must be clearly visible
- Ensure model is trained on similar helmet types

### Failed Plate Detection
- Check ENABLE_IMAGE_PLATE_OCR=1
- Verify OCR reader initialized
- Plates must be clear and readable
- Verify Indian plate format is supported

### No GPU Acceleration
- EasyOCR defaults to CPU (gpu=False in code)
- YOLO auto-detects GPU availability
- Check CUDA compatibility if manual GPU setup needed

---

## Testing the System

### 1. Test Helmet Detection
```bash
# Upload image with helmet
POST /detect_image
# Expected: helmet_status: "Yes"
```

### 2. Test Plate Detection
```bash
# Upload clear plate image
POST /detect_image
# Expected: plate_numbers: ["XX##XX####"]
```

### 3. Test Performance
```bash
# Upload 10 same images
# First: ~300ms, Rest: ~5ms (from cache)
```

---

## File Structure
```
capstone project frontend and backend/
├── app.py                           # Main Flask application
├── requirements.txt                 # Python dependencies
├── .env.example                     # Configuration template
├── helmet_model.pt                  # Helmet detection model
├── plate_model.pt                   # Plate detection model
├── traffic_model.pt                 # Traffic detection model
├── static/
│   ├── uploads/                     # Uploaded files
│   ├── results/                     # Detection results
│   └── style.css                    # Styling
└── templates/
    ├── index.html                   # Main page
    ├── image.html                   # Image detection page
    └── webcam.html                  # Webcam page
```

---

## Dependencies
- **Flask**: Web framework
- **OpenCV**: Image processing
- **EasyOCR**: Character recognition
- **YOLOv8**: Object detection
- **NumPy**: Numerical operations

Install: `pip install -r requirements.txt`

---

## Quick Start

1. **Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure** (Optional)
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

3. **Place Models**
   - `helmet_model.pt` → project root
   - `plate_model.pt` → project root (optional)

4. **Run**
   ```bash
   python app.py
   ```

5. **Access**
   - Open: http://localhost:5004
   - Upload image or video
   - View results

---

## Notes
- All responses include `helmet_detection_supported: true/false`
- Plate numbers are deduplicated (no duplicates in results)
- Images are cached for 32 most recent uploads
- Video processing samples every 8th frame (configurable)
