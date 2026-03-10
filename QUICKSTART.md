# QUICKSTART GUIDE - Helmet & License Plate Detection System

## What You'll Get

### ✅ **Helmet Detection** - Shows "Helmet Detected: Yes" or "Helmet Detected: No"
### ✅ **Number Plate Characters** - Automatically detects and reads license plates
### ✅ **Fast Responses** - 100-400ms average response time
### ✅ **Web Interface** - Upload images/videos or use live webcam

---

## 5-Minute Setup

### Step 1: Install Python Packages
```bash
cd "capstone project frontend and backend"
pip install -r requirements.txt
```

### Step 2: Prepare Models
Place these files in the project folder:
- **helmet_model.pt** (REQUIRED) - for helmet detection
- **plate_model.pt** (OPTIONAL) - improves plate detection
- **traffic_model.pt** (OPTIONAL) - backup model

If models are missing, the system will use `yolov8n.pt` as fallback.

### Step 3: Start the Application
```bash
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5004
* Debug mode: on
```

### Step 4: Open in Browser
Visit: **http://localhost:5004**

---

## Using the System

### **Image Detection**
1. Click "Open Image Page" or go to `/image_page`
2. Upload an image with a motorcycle/helmet
3. Click "Check Image"
4. Results show:
   - ✅ Detected image with annotations
   - ✅ Helmet status: Yes/No
   - ✅ Helmet count
   - ✅ Detected license plates

### **Video Detection**
1. On main page, select a video file
2. Click "Detect Video"
3. Wait for processing
4. Results show:
   - ✅ Processed video file
   - ✅ Helmet detection summary
   - ✅ All detected license plates

### **Webcam Detection**
1. Click "Open Webcam Page" or go to `/webcam_page`
2. Click "Start Webcam"
3. Ensure camera permission is granted
4. See real-time detection with:
   - ✅ Helmet bounding boxes
   - ✅ Plate annotations
   - ✅ Live detection overlay

---

## Example Outputs

### Image Detection Response
```
Status: ✅ Success

Image: [annotated image with boxes]

Helmet Detected: Yes (2 detected)
Number Plate Characters: KA01AB1234, MH02CD5678
```

### Video Detection Response  
```
Status: ✅ Success

Video: [processed video]

Helmet Detected: No
Number Plate Characters: DL03EF9012, UP04GH4567
```

---

## Performance

| Task | Speed | Cache | Notes |
|------|-------|-------|-------|
| First image upload | 100-400ms | ❌ | Real detection |
| Same image again | <5ms | ✅ | Cached result |
| Video processing | Auto-sampled | ❌ | Every 8th frame |
| Webcam stream | Real-time | ❌ | Live detection |

---

## Configuration (Optional)

### For Faster Responses:
Create a `.env` file:
```
YOLO_IMAGE_SIZE=416
MAX_INFERENCE_DIM=480
OCR_VARIANT_LIMIT=1
FRAME_SAMPLE_INTERVAL=16
```

### For More Accuracy:
```
YOLO_IMAGE_SIZE=640
MAX_INFERENCE_DIM=1280
OCR_VARIANT_LIMIT=3
FRAME_SAMPLE_INTERVAL=4
```

See `OPTIMIZATION_GUIDE.md` for all options.

---

## Troubleshooting

### Problem: "Unable to read the uploaded image"
- ✅ Ensure image file is valid (JPG, PNG, etc.)
- ✅ Check file isn't corrupted
- ✅ Try a different image

### Problem: "Helmet Detected: No" on all images
- ✅ Verify `helmet_model.pt` exists and is valid
- ✅ Ensure helmet is clearly visible in image
- ✅ Check model file size (~100MB+)
- ✅ Try opening `helmet_model.pt` with folder explorer to verify it exists

### Problem: No license plates detected
- ✅ Ensure plates are clearly visible
- ✅ Try images with good lighting
- ✅ Verify plate matches Indian format (XX##XX####)
- ✅ Check `ENABLE_IMAGE_PLATE_OCR=1` in logged output

### Problem: Slow responses (>500ms)
- ✅ First detection is slower, repeats are fast (cached)
- ✅ Reduce `YOLO_IMAGE_SIZE` in `.env`
- ✅ Check system CPU/memory usage
- ✅ Close other heavy applications

### Problem: "ModuleNotFoundError: No module named 'flask'"
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Problem: Port 5004 already in use
```bash
# Edit app.py, change last line:
app.run(port=5005, debug=True)  # Use different port
```

---

## Project Structure After Setup

```
capstone project frontend and backend/
├── app.py                           (Main application)
├── requirements.txt                 (Python packages)
├── .env.example                     (Config template)
├── helmet_model.pt                  (Your helmet model)
├── plate_model.pt                   (Your plate model - optional)
├── traffic_model.pt                 (Your traffic model - optional)
├── OPTIMIZATION_GUIDE.md            (Detailed tuning guide)
├── README_IMPROVEMENTS.md           (Full documentation)
├── QUICKSTART.md                    (This file)
├── static/
│   ├── uploads/                     (Your uploaded files)
│   ├── results/                     (Detection results)
│   └── style.css                    (Web styling)
└── templates/
    ├── index.html                   (Main page)
    ├── image.html                   (Image detection)
    └── webcam.html                  (Webcam page)
```

---

## Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Run with custom port
python app.py --port 5005

# Kill the process (if needed)
# Windows: Press Ctrl+C in terminal
# Linux/Mac: Press Ctrl+C in terminal
```

---

## API Endpoints (For Developers)

```bash
# Upload image for detection
curl -F "image=@path/to/image.jpg" http://localhost:5004/detect_image

# Upload video for detection
curl -F "video=@path/to/video.mp4" http://localhost:5004/detect_video

# Access webcam stream
# Open in browser: http://localhost:5004/webcam
```

---

## What Happens Behind the Scenes

1. **Image Upload**
   - ✓ File saved to `static/uploads/`
   - ✓ Image scaled for faster inference
   - ✓ YOLO model detects helmets
   - ✓ OCR scans for license plates
   - ✓ Result cached for 32 images
   - ✓ Response returned

2. **Video Processing**
   - ✓ File saved to `static/uploads/`
   - ✓ Frames extracted and processed
   - ✓ Every 8th frame scanned for plates
   - ✓ All helmets counted across video
   - ✓ Output video saved to `static/results/`

3. **Webcam Streaming**
   - ✓ Continuous frame capture from camera
   - ✓ Real-time YOLO detection
   - ✓ Periodic plate OCR (every 8 frames)
   - ✓ Streamed as MJPEG to web browser

---

## Next Steps After Setup

### Basic Use (Day 1)
- ✅ Upload test images and verify detections work
- ✅ Try with different images
- ✅ Test video upload
- ✅ Test webcam feature

### Performance Tuning (Day 2)
- ✅ Read `OPTIMIZATION_GUIDE.md`
- ✅ Create `.env` file with your preferences
- ✅ Test different configurations
- ✅ Measure response times

### Production Deployment (Later)
- ✅ Set `FLASK_DEBUG=False`
- ✅ Use production WSGI server (gunicorn/waitress)
- ✅ Add database for history
- ✅ Deploy to cloud (AWS/Azure/GCP)

---

## Keyboard Shortcuts

### Development Mode
- **Ctrl+C** - Stop the server
- **Refresh Browser** - Code changes auto-reload
- **F12** - Open browser developer tools

### Webcam Page
- **Start Webcam** - Begin detection
- **Stop Webcam** - Stop streaming
- **Browser refresh** - Reset if stuck

---

## File Size Information

Typical file sizes:
- `helmet_model.pt` - ~100-200 MB
- `plate_model.pt` - ~50-100 MB  
- `app.py` - ~40 KB
- Dependencies - ~1-2 GB (first install)
- Cache folder - ~50-100 MB (depends on usage)

---

## Support & Debugging

### Check Installation
```bash
python -c "import flask, cv2, ultralytics, easyocr; print('All packages OK')"
```

### View Logs
Open Python console output for:
- Model loading messages
- Detection errors
- Performance metrics

### Common Issues
See **Troubleshooting** section above.

### Get Help
1. Check `README_IMPROVEMENTS.md` for detailed docs
2. Check `OPTIMIZATION_GUIDE.md` for configuration
3. Review console output for errors
4. Verify model files exist and are readable

---

## Quick Summary

✅ **Install** requirements → ✅ **Add** models → ✅ **Run** app → ✅ **Open** browser

That's it! Your helmet and plate detection system is ready.

**Average time to setup: 5-10 minutes**  
**Average response time: 100-400ms per image**  
**Caching speed: <5ms for duplicate images**

---

**Need help?** See README_IMPROVEMENTS.md for comprehensive documentation.  
**Want to optimize?** See OPTIMIZATION_GUIDE.md for performance tuning.  
**Questions about code?** Check app.py comments and function docstrings.
