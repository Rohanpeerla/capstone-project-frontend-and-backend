from flask import Flask, render_template, request, jsonify, Response, url_for
import easyocr
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import hashlib
import numpy as np
import os
import re
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used\."
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELMET_MODEL_PATH = os.environ.get(
    "HELMET_MODEL_PATH",
    os.path.join(BASE_DIR, "helmet_model.pt")
)
PLATE_MODEL_PATH = os.environ.get(
    "PLATE_MODEL_PATH",
    os.path.join(BASE_DIR, "plate_model.pt")
)
TRAFFIC_MODEL_PATH = os.environ.get(
    "TRAFFIC_MODEL_PATH",
    os.path.join(BASE_DIR, "traffic_model.pt")
)
DEFAULT_MODEL = os.environ.get("YOLO_DEFAULT_MODEL", "yolov8n.pt")
FRAME_SAMPLE_INTERVAL = max(1, int(os.environ.get("FRAME_SAMPLE_INTERVAL", "8")))
MAX_VEHICLE_CROPS = max(1, int(os.environ.get("MAX_VEHICLE_CROPS", "1")))
OCR_VARIANT_LIMIT = max(1, int(os.environ.get("OCR_VARIANT_LIMIT", "1")))
MAX_INFERENCE_DIM = max(320, int(os.environ.get("MAX_INFERENCE_DIM", "480")))
YOLO_IMAGE_SIZE = max(320, int(os.environ.get("YOLO_IMAGE_SIZE", "320")))
ENABLE_IMAGE_PLATE_OCR = os.environ.get("ENABLE_IMAGE_PLATE_OCR", "0") == "1"
ENABLE_VIDEO_PLATE_OCR = os.environ.get("ENABLE_VIDEO_PLATE_OCR", "0") == "1"
ENABLE_WEBCAM_PLATE_OCR = os.environ.get("ENABLE_WEBCAM_PLATE_OCR", "0") == "1"
ENABLE_IMAGE_FULL_FRAME_FALLBACK = os.environ.get("ENABLE_IMAGE_FULL_FRAME_FALLBACK", "0") == "1"
MAX_PLATE_REGION_PROPOSALS = max(1, int(os.environ.get("MAX_PLATE_REGION_PROPOSALS", "3")))
IMAGE_CACHE_SIZE = max(1, int(os.environ.get("IMAGE_CACHE_SIZE", "32")))
helmet_model = None
plate_model = None
ocr_reader = None
helmet_support_cache = None
image_result_cache = {}

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
HELMET_CLASS_NAMES = {"helmet", "helmets"}
PLATE_CLASS_NAMES = {"number plate", "number plates", "license plate", "licence plate", "plate"}
PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{1,4}$"),
]
ALPHA_TO_DIGIT_MAP = str.maketrans({
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
})
DIGIT_TO_ALPHA_MAP = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
})


def resize_for_inference(frame):
    if frame is None or frame.size == 0:
        return frame

    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= MAX_INFERENCE_DIM:
        return frame

    scale = MAX_INFERENCE_DIM / float(longest_side)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def load_image_from_path(path):
    if not os.path.exists(path):
        return None

    file_bytes = np.fromfile(path, dtype=np.uint8)
    if file_bytes.size == 0:
        return None

    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def get_file_hash(path):
    if not os.path.exists(path):
        return None

    file_bytes = np.fromfile(path, dtype=np.uint8)
    if file_bytes.size == 0:
        return None

    return hashlib.sha256(file_bytes.tobytes()).hexdigest()


def get_helmet_model():
    global helmet_model
    if helmet_model is None:
        if os.path.exists(HELMET_MODEL_PATH):
            try:
                helmet_model = YOLO(HELMET_MODEL_PATH)
            except Exception as e:
                print(f"Error loading helmet model: {e}")
                helmet_model = YOLO(DEFAULT_MODEL)
        else:
            helmet_model = YOLO(DEFAULT_MODEL)
    return helmet_model


def get_plate_model():
    global plate_model
    if plate_model is None:
        if os.path.exists(PLATE_MODEL_PATH):
            plate_model = YOLO(PLATE_MODEL_PATH)
    return plate_model


def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return ocr_reader


def warm_up_runtime():
    get_helmet_model()
    get_plate_model()
    helmet_detection_supported()
    if ENABLE_IMAGE_PLATE_OCR or ENABLE_VIDEO_PLATE_OCR or ENABLE_WEBCAM_PLATE_OCR:
        get_ocr_reader()


def get_cached_image_result(cache_key):
    cached = image_result_cache.get(cache_key)
    if cached is None:
        return None
    return dict(cached)


def set_cached_image_result(cache_key, payload):
    if cache_key is None:
        return

    if len(image_result_cache) >= IMAGE_CACHE_SIZE:
        oldest_key = next(iter(image_result_cache))
        image_result_cache.pop(oldest_key, None)

    image_result_cache[cache_key] = dict(payload)


def get_model_status():
    return {
        "helmet_model_ready": os.path.exists(HELMET_MODEL_PATH),
        "plate_model_ready": os.path.exists(PLATE_MODEL_PATH),
        "traffic_model_ready": os.path.exists(TRAFFIC_MODEL_PATH),
        "helmet_detection_supported": helmet_detection_supported()
    }


def helmet_detection_supported():
    global helmet_support_cache
    if helmet_support_cache is None:
        model = get_helmet_model()
        helmet_support_cache = any(name in HELMET_CLASS_NAMES for name in model.names.values())
    return helmet_support_cache


def is_helmet_class(class_name):
    return class_name in HELMET_CLASS_NAMES


def is_plate_class(class_name):
    return class_name in PLATE_CLASS_NAMES


def clean_plate_text(text):
    cleaned = normalize_indian_plate_text(text)
    if cleaned:
        return cleaned

    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned) < 5:
        return None
    if len(cleaned) > 12:
        return None
    if not any(char.isdigit() for char in cleaned):
        return None
    if not any(char.isalpha() for char in cleaned):
        return None
    if cleaned[0].isdigit():
        return None
    if not any(pattern.fullmatch(cleaned) for pattern in PLATE_PATTERNS):
        return None
    return cleaned


def normalize_indian_plate_text(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned) < 8:
        return None

    for series_length in (2, 1, 3):
        required_length = 2 + 2 + series_length + 4
        if len(cleaned) < required_length:
            continue

        state = cleaned[0:2].translate(DIGIT_TO_ALPHA_MAP)
        district = cleaned[2:4].translate(ALPHA_TO_DIGIT_MAP)
        series = cleaned[4:4 + series_length].translate(DIGIT_TO_ALPHA_MAP)
        number = cleaned[4 + series_length:].translate(ALPHA_TO_DIGIT_MAP)

        candidate = f"{state}{district}{series}{number[:4]}"
        if any(pattern.fullmatch(candidate) for pattern in PLATE_PATTERNS):
            return candidate

    return None


def generate_plate_variants(image):
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    bilateral = cv2.bilateralFilter(upscaled, 9, 15, 15)
    adaptive = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    otsu = cv2.threshold(
        bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return [upscaled, adaptive, otsu][:OCR_VARIANT_LIMIT]


def read_plate_candidates(image, fast_mode=False):
    reader = get_ocr_reader()
    candidates = []
    
    # In fast mode, use fewer variants for quicker response
    variant_limit = 1 if fast_mode else OCR_VARIANT_LIMIT

    for variant in generate_plate_variants(image)[:variant_limit]:
        ocr_results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            batch_size=1
        )
        sorted_results = sorted(
            ocr_results,
            key=lambda item: (
                min(point[1] for point in item[0]),
                min(point[0] for point in item[0]),
            )
        )

        combined_text = "".join(raw_text for _, raw_text, _ in sorted_results)
        combined_confidence = (
            sum(float(confidence) for _, _, confidence in sorted_results) / len(sorted_results)
            if sorted_results else 0.0
        )
        combined_plate_text = clean_plate_text(combined_text)
        if combined_plate_text:
            candidates.append((combined_plate_text, combined_confidence + 0.25))

        for _, raw_text, confidence in sorted_results:
            plate_text = clean_plate_text(raw_text)
            if not plate_text:
                continue
            candidates.append((plate_text, confidence))

    candidates.sort(key=lambda item: (len(item[0]), item[1]), reverse=True)
    return candidates


def build_plate_crops(vehicle_crop):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return []

    crop_height, crop_width = vehicle_crop.shape[:2]
    return [
        vehicle_crop[int(crop_height * 0.45):crop_height, :],
        vehicle_crop[int(crop_height * 0.55):crop_height, int(crop_width * 0.10):int(crop_width * 0.90)],
    ]


def get_box_area(box):
    x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
    return max(0, x2 - x1) * max(0, y2 - y1)


def get_rect_area(rect):
    _, _, width, height = rect
    return max(0, width) * max(0, height)


def add_plate_result(plate_numbers, seen, text):
    if text and text not in seen:
        seen.add(text)
        plate_numbers.append(text)
        return True
    return False


def annotate_plate_text(frame, x, y, text):
    cv2.putText(
        frame,
        text,
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


def find_plate_like_regions(image):
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    image_height, image_width = image.shape[:2]
    min_area = image_height * image_width * 0.002
    max_area = image_height * image_width * 0.25

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue

        area = width * height
        aspect_ratio = width / float(height)
        if area < min_area or area > max_area:
            continue
        if aspect_ratio < 2.0 or aspect_ratio > 8.0:
            continue

        pad_x = int(width * 0.08)
        pad_y = int(height * 0.20)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image_width, x + width + pad_x)
        y2 = min(image_height, y + height + pad_y)
        regions.append((x1, y1, x2, y2))

    regions.sort(key=get_rect_area, reverse=True)

    deduped = []
    for region in regions:
        x1, y1, x2, y2 = region
        duplicate = False
        for existing in deduped:
            ex1, ey1, ex2, ey2 = existing
            if abs(x1 - ex1) < 15 and abs(y1 - ey1) < 15 and abs(x2 - ex2) < 15 and abs(y2 - ey2) < 15:
                duplicate = True
                break
        if not duplicate:
            deduped.append(region)
        if len(deduped) >= MAX_PLATE_REGION_PROPOSALS:
            break

    return deduped


def detect_plate_text_in_crop(image, fast_mode=False):
    best_text = None
    best_score = -1.0

    for plate_text, confidence in read_plate_candidates(image, fast_mode=fast_mode):
        score = len(plate_text) + confidence + 0.75
        if score > best_score:
            best_text = plate_text
            best_score = score

    if not fast_mode:
        for plate_crop in build_plate_crops(image):
            for plate_text, confidence in read_plate_candidates(plate_crop, fast_mode=fast_mode):
                score = len(plate_text) + confidence
                if score > best_score:
                    best_text = plate_text
                    best_score = score

        for x1, y1, x2, y2 in find_plate_like_regions(image):
            region_crop = image[y1:y2, x1:x2]
            if region_crop.size == 0:
                continue
            for plate_text, confidence in read_plate_candidates(region_crop, fast_mode=fast_mode):
                score = len(plate_text) + confidence + 0.5
                if score > best_score:
                    best_text = plate_text
                    best_score = score

    return best_text, best_score


def add_full_frame_plate_fallback(frame, annotated, plate_numbers, seen, fast_mode=False):
    frame_height, frame_width = frame.shape[:2]
    search_regions = [
        (0, int(frame_height * 0.45), frame_width, frame_height),
        (0, int(frame_height * 0.55), frame_width, frame_height),
    ]

    for x1, y1, x2, y2 in search_regions:
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            continue

        best_text, _ = detect_plate_text_in_crop(region, fast_mode=fast_mode)
        if add_plate_result(plate_numbers, seen, best_text):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
            annotate_plate_text(annotated, x1, y1, best_text)
            return


def detect_plates_fast(frame):
    """Fast plate detection - just read text from the full image."""
    if frame is None or frame.size == 0:
        return []
    
    reader = get_ocr_reader()
    plate_numbers = []
    seen = set()
    
    # Single pass OCR on full frame
    ocr_results = reader.readtext(
        frame,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        batch_size=1
    )
    
    sorted_results = sorted(
        ocr_results,
        key=lambda item: (
            min(point[1] for point in item[0]),
            min(point[0] for point in item[0]),
        )
    )

    # Try combined text first
    combined_text = "".join(raw_text for _, raw_text, _ in sorted_results)
    combined_plate = clean_plate_text(combined_text)
    if combined_plate and combined_plate not in seen:
        plate_numbers.append(combined_plate)
        seen.add(combined_plate)

    # Try individual results
    for _, raw_text, confidence in sorted_results:
        plate_text = clean_plate_text(raw_text)
        if plate_text and plate_text not in seen:
            plate_numbers.append(plate_text)
            seen.add(plate_text)
    
    return plate_numbers


def detect_image_plate_numbers(original_frame):
    custom_plate_model = get_plate_model()
    if custom_plate_model is not None and custom_plate_model is not get_helmet_model():
        return extract_number_plates(original_frame, fast_mode=True)
    return detect_plates_fast(original_frame)


def extract_number_plates_from_vehicle_boxes(frame, detection_result, fast_mode=False):
    plate_numbers = []
    seen = set()
    annotated = detection_result.plot()
    vehicle_boxes = []

    for box in detection_result.boxes:
        class_name = detection_result.names[int(box.cls[0])]
        if class_name not in VEHICLE_CLASSES:
            continue
        vehicle_boxes.append(box)

    vehicle_boxes.sort(key=get_box_area, reverse=True)

    for box in vehicle_boxes[:MAX_VEHICLE_CROPS]:
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            continue

        best_text, _ = detect_plate_text_in_crop(vehicle_crop, fast_mode=fast_mode)

        if add_plate_result(plate_numbers, seen, best_text):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            annotate_plate_text(annotated, x1, y1, best_text)

    if ENABLE_IMAGE_PLATE_OCR and ENABLE_IMAGE_FULL_FRAME_FALLBACK and not plate_numbers:
        add_full_frame_plate_fallback(frame, annotated, plate_numbers, seen)

    return annotated, plate_numbers


def extract_number_plates(frame, helmet_detection_result=None, fast_mode=False):
    custom_plate_model = get_plate_model()
    if custom_plate_model is not None and custom_plate_model is not get_helmet_model():
        results = custom_plate_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)
        annotated = frame.copy()
        plate_numbers = []
        seen = set()

        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            best_text, _ = detect_plate_text_in_crop(plate_crop, fast_mode=fast_mode)

            if add_plate_result(plate_numbers, seen, best_text):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                annotate_plate_text(annotated, x1, y1, best_text)

        if ENABLE_IMAGE_PLATE_OCR and ENABLE_IMAGE_FULL_FRAME_FALLBACK and not plate_numbers:
            add_full_frame_plate_fallback(frame, annotated, plate_numbers, seen, fast_mode=fast_mode)

        return annotated, plate_numbers

    if helmet_detection_result is None:
        helmet_detection_result = get_helmet_model()(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)[0]

    matching_plate_boxes = [
        box for box in helmet_detection_result.boxes
        if is_plate_class(helmet_detection_result.names[int(box.cls[0])])
    ]
    if matching_plate_boxes:
        annotated = helmet_detection_result.plot()
        plate_numbers = []
        seen = set()

        for box in matching_plate_boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            best_text, _ = detect_plate_text_in_crop(plate_crop, fast_mode=fast_mode)

            if add_plate_result(plate_numbers, seen, best_text):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                annotate_plate_text(annotated, x1, y1, best_text)

        return annotated, plate_numbers

    return extract_number_plates_from_vehicle_boxes(frame, helmet_detection_result, fast_mode=fast_mode)


@app.route('/')
def home():
    model_status = get_model_status()
    return render_template(
        "index.html",
        model_status=model_status
    )


@app.route('/webcam_page')
def webcam_page():
    model_status = get_model_status()
    return render_template(
        "webcam.html",
        model_status=model_status
    )


@app.route('/image_page')
def image_page():
    model_status = get_model_status()
    return render_template(
        "image.html",
        model_status=model_status
    )


@app.route('/detect_image', methods=['POST'])
def detect_image():
    current_model = get_helmet_model()

    file = request.files.get('image')
    if file is None or not file.filename:
        return jsonify({"error": "No image file was uploaded."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid image filename."}), 400

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    cache_key = get_file_hash(path)
    cached_payload = get_cached_image_result(cache_key)
    if cached_payload is not None:
        return jsonify(cached_payload)

    original_frame = load_image_from_path(path)
    if original_frame is None:
        return jsonify({"error": "Unable to read the uploaded image."}), 400

    frame = resize_for_inference(original_frame)
    results = current_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)
    result_filename = "result_" + filename
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    annotated_frame = results[0].plot()
    plate_numbers = []

    # Detect plates using simple fast method
    if ENABLE_IMAGE_PLATE_OCR:
        try:
            plate_numbers = detect_plates_fast(original_frame)
        except Exception as e:
            print(f"Plate detection error: {e}")

    cv2.imwrite(result_path, annotated_frame)

    classes = [current_model.names[int(c)] for c in results[0].boxes.cls]
    helmet_detected = any(is_helmet_class(class_name) for class_name in classes)
    helmet_status = "Yes" if helmet_detected else "No"
    
    helmet_count = sum(1 for class_name in classes if is_helmet_class(class_name))

    payload = {
        "image_path": url_for("static", filename=f"results/{result_filename}"),
        "helmet_status": helmet_status,
        "helmet_detected": helmet_detected,
        "helmet_count": helmet_count,
        "plate_numbers": plate_numbers,
        "detection_summary": f"Detection completed. Helmet: {helmet_status}" + (f", Plates: {', '.join(plate_numbers)}" if plate_numbers else ""),
        "helmet_detection_supported": helmet_detection_supported()
    }
    set_cached_image_result(cache_key, payload)
    return jsonify(payload)


@app.route('/detect_video', methods=['POST'])
def detect_video():
    current_model = get_helmet_model()

    file = request.files['video']
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    cap = cv2.VideoCapture(path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_filename = "video_" + str(int(time.time())) + ".mp4"
    out_path = os.path.join(RESULT_FOLDER, out_filename)

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    helmet_detected = False
    helmet_count = 0
    detected_plates = set()

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame = resize_for_inference(frame)
        results = current_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)

        for r in results:
            for cls in r.boxes.cls:
                class_name = current_model.names[int(cls)]
                if is_helmet_class(class_name):
                    helmet_detected = True
                    helmet_count += 1

        if ENABLE_VIDEO_PLATE_OCR and frame_index % FRAME_SAMPLE_INTERVAL == 0:
            annotated_frame, plate_numbers = extract_number_plates(frame, results[0])
            detected_plates.update(plate_numbers)
        else:
            annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    helmet_status = "Yes" if helmet_detected else "No"

    return jsonify({
        "video_path": url_for("static", filename=f"results/{out_filename}"),
        "helmet_status": helmet_status,
        "helmet_detected": helmet_detected,
        "helmet_count": helmet_count,
        "detection_summary": f"Helmet Detected: {helmet_status}" + (f" ({helmet_count} detections)" if helmet_detected else ""),
        "plate_numbers": sorted(detected_plates),
        "helmet_detection_supported": helmet_detection_supported()
    })


@app.route('/webcam')
def webcam():
    def generate():
        current_model = get_helmet_model()

        cap = cv2.VideoCapture(0)
        frame_index = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1
            frame = resize_for_inference(frame)
            results = current_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)
            if ENABLE_WEBCAM_PLATE_OCR and frame_index % FRAME_SAMPLE_INTERVAL == 0:
                frame, _ = extract_number_plates(frame, results[0])
            else:
                frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )

        cap.release()

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    warm_up_runtime()
    app.run(port = 5004, debug=True)
