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
HELMET_CONFIDENCE_THRESHOLD = float(os.environ.get("HELMET_CONFIDENCE_THRESHOLD", "62"))
DIRECT_HELMET_CONFIDENCE_THRESHOLD = float(os.environ.get("DIRECT_HELMET_CONFIDENCE_THRESHOLD", "85"))
DIRECT_NO_HELMET_CONFIDENCE_THRESHOLD = float(os.environ.get("DIRECT_NO_HELMET_CONFIDENCE_THRESHOLD", "60"))
IMAGE_RESULT_VERSION = "helmet-v8"
helmet_model = None
plate_model = None
ocr_reader = None
helmet_support_cache = None
face_cascade = None
image_result_cache = {}

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
HELMET_CLASS_NAMES = {"helmet", "helmets", "with helmet", "with_helmet", "helmet_present"}
NO_HELMET_CLASS_NAMES = {"no helmet", "no_helmet", "without helmet", "without_helmet", "helmetless"}
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


def get_face_cascade():
    global face_cascade
    if face_cascade is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            classifier = cv2.CascadeClassifier(cascade_path)
            face_cascade = classifier if not classifier.empty() else False
        else:
            face_cascade = False
    return face_cascade if face_cascade is not False else None


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
        helmet_support_cache = any(
            str(name).strip().lower() in HELMET_CLASS_NAMES
            for name in model.names.values()
        )
    return helmet_support_cache


def is_helmet_class(class_name):
    return str(class_name).strip().lower() in HELMET_CLASS_NAMES


def is_no_helmet_class(class_name):
    return str(class_name).strip().lower() in NO_HELMET_CLASS_NAMES


def assess_head_coverage(person_crop):
    if person_crop is None or person_crop.size == 0:
        return {
            "face_visible": False,
            "shell_like": False,
            "top_dark_ratio": 0.0,
            "side_dark_ratio": 0.0,
            "skin_ratio": 1.0,
        }

    height, width = person_crop.shape[:2]
    if height < 60 or width < 40:
        return {
            "face_visible": False,
            "shell_like": False,
            "top_dark_ratio": 0.0,
            "side_dark_ratio": 0.0,
            "skin_ratio": 1.0,
        }

    head_region_height = max(1, int(height * 0.42))
    head_region = person_crop[:head_region_height, :]
    gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    h, w = head_region.shape[:2]
    center_x_start = max(0, int(w * 0.15))
    center_x_end = min(w, int(w * 0.85))

    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 25, 60], dtype=np.uint8)
    upper_skin = np.array([30, 180, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = cv2.countNonZero(skin_mask) / skin_mask.size if skin_mask.size > 0 else 1.0

    left_band = gray[:, :max(1, int(w * 0.18))]
    right_band = gray[:, min(w - 1, int(w * 0.82)):] if w > 1 else gray
    top_band = gray[:max(1, int(h * 0.25)), :]
    side_dark = cv2.countNonZero(cv2.threshold(left_band, 135, 255, cv2.THRESH_BINARY_INV)[1])
    side_dark += cv2.countNonZero(cv2.threshold(right_band, 135, 255, cv2.THRESH_BINARY_INV)[1])
    side_area = max(1, left_band.size + right_band.size)
    side_dark_ratio = side_dark / side_area
    top_dark = cv2.countNonZero(cv2.threshold(top_band, 145, 255, cv2.THRESH_BINARY_INV)[1])
    top_dark_ratio = top_dark / top_band.size if top_band.size > 0 else 0.0

    face_visible = False
    face_detector = get_face_cascade()
    if face_detector is not None:
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        head_area = float(max(1, h * w))
        for fx, fy, fw, fh in faces:
            face_area_ratio = (fw * fh) / head_area
            face_center_x = fx + (fw / 2.0)
            lower_face_visible = (fy + fh) > (h * 0.55)
            if face_area_ratio >= 0.08 and center_x_start <= face_center_x <= center_x_end and lower_face_visible:
                face_visible = True
                break

    shell_like = (
        top_dark_ratio > 0.19 or
        side_dark_ratio > 0.22 or
        skin_ratio < 0.30
    )

    return {
        "face_visible": face_visible,
        "shell_like": shell_like,
        "top_dark_ratio": top_dark_ratio,
        "side_dark_ratio": side_dark_ratio,
        "skin_ratio": skin_ratio,
    }


def detect_helmet_in_person_crop(person_crop):
    """
    Detect helmet in a person crop using multi-method analysis.
    Handles various crop sizes and image qualities.
    Returns: (helmet_detected: bool, confidence: float 0-100)
    """
    if person_crop is None or person_crop.size == 0:
        return False, 0.0
    
    try:
        height = person_crop.shape[0]
        width = person_crop.shape[1]
        if height < 80 or width < 40:
            return False, 0.0

        head_region_height = max(1, int(height * 0.42))
        head_region = person_crop[:head_region_height, :]
        
        if head_region.size < 100 or head_region.shape[0] < 20 or head_region.shape[1] < 20:
            return False, 0.0
        
        # Normalize crop size for threshold adjustment
        crop_area = person_crop.shape[0] * person_crop.shape[1]
        is_small_crop = crop_area < 100000  # Less than ~316x316
        is_large_crop = crop_area > 500000  # Greater than ~707x707
        
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        h, w = head_region.shape[:2]
        center_x_start = max(0, int(w * 0.15))
        center_x_end = min(w, int(w * 0.85))
        face_detector = get_face_cascade()
        detected_face = None
        if face_detector is not None:
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24)
            )
            head_area = float(head_region.shape[0] * head_region.shape[1])
            for fx, fy, fw, fh in faces:
                face_area_ratio = (fw * fh) / head_area if head_area > 0 else 0.0
                face_center_x = fx + (fw / 2.0)
                if face_area_ratio >= 0.08 and center_x_start <= face_center_x <= center_x_end:
                    detected_face = (fx, fy, fw, fh, face_area_ratio)
                    break
        
        # Method 1: Local contrast analysis
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        contrast = cv2.absdiff(gray, blurred)
        contrast_mean = np.mean(contrast)
        
        # Method 2: Edge detection
        edges = cv2.Canny(gray, 30, 150)
        edge_count = cv2.countNonZero(edges)
        edge_density = edge_count / edges.size if edges.size > 0 else 0
        
        # Method 3: Dark regions (helmet indicator)
        _, binary_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        dark_count = cv2.countNonZero(binary_dark)
        dark_ratio = dark_count / binary_dark.size if binary_dark.size > 0 else 0
        
        # Method 4: Medium-dark regions (better for helmets)
        _, binary_medium = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        medium_count = cv2.countNonZero(binary_medium)
        medium_ratio = medium_count / binary_medium.size if binary_medium.size > 0 else 0

        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 25, 60], dtype=np.uint8)
        upper_skin = np.array([30, 180, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = cv2.countNonZero(skin_mask) / skin_mask.size if skin_mask.size > 0 else 0

        left_band = gray[:, :max(1, int(w * 0.18))]
        right_band = gray[:, min(w - 1, int(w * 0.82)):] if w > 1 else gray
        top_band = gray[:max(1, int(h * 0.25)), :]
        side_dark = cv2.countNonZero(cv2.threshold(left_band, 135, 255, cv2.THRESH_BINARY_INV)[1])
        side_dark += cv2.countNonZero(cv2.threshold(right_band, 135, 255, cv2.THRESH_BINARY_INV)[1])
        side_area = left_band.size + right_band.size if (left_band.size + right_band.size) > 0 else 1
        side_dark_ratio = side_dark / side_area
        top_dark = cv2.countNonZero(cv2.threshold(top_band, 145, 255, cv2.THRESH_BINARY_INV)[1])
        top_dark_ratio = top_dark / top_band.size if top_band.size > 0 else 0
        
        # Method 5: Center region analysis
        center_region = gray[:, center_x_start:center_x_end]
        center_dark = cv2.countNonZero(
            cv2.threshold(center_region, 120, 255, cv2.THRESH_BINARY_INV)[1]
        )
        center_dark_ratio = center_dark / center_region.size if center_region.size > 0 else 0
        
        # Method 6: Morphological shape analysis
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary_dark, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        helmet_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 30:
                x, y, cw, ch = cv2.boundingRect(contour)
                if y < h * 0.7 and center_x_start <= x + cw/2 <= center_x_end:
                    helmet_contours += 1
        
        # Adaptive thresholds based on crop size - balanced for both detection and false positives
        if is_small_crop:
            threshold_edge = 0.08
            threshold_dark = 0.24
            threshold_contrast = 10.0
            threshold_medium = 0.18
        elif is_large_crop:
            threshold_edge = 0.12
            threshold_dark = 0.30
            threshold_contrast = 26
            threshold_medium = 0.26
        else:
            threshold_edge = 0.11
            threshold_dark = 0.27
            threshold_contrast = 20
            threshold_medium = 0.22
        
        # Signal voting system with strength scores
        signal_strengths = []
        
        # Signal 1: Edges present (0-100)
        edge_strength = min(100, (edge_density / threshold_edge) * 100) if threshold_edge > 0 else 0
        signal_strengths.append(edge_strength)
        
        # Signal 2: Dark regions (0-100)
        dark_strength = min(100, (dark_ratio / threshold_dark) * 100) if threshold_dark > 0 else 0
        signal_strengths.append(dark_strength)
        
        # Signal 3: Medium-dark regions (0-100)
        medium_strength = min(100, (medium_ratio / threshold_medium) * 100) if threshold_medium > 0 else 0
        signal_strengths.append(medium_strength)
        
        # Signal 4: Contrast present (0-100)
        contrast_strength = min(100, (contrast_mean / threshold_contrast) * 100) if threshold_contrast > 0 else 0
        signal_strengths.append(contrast_strength)
        
        # Signal 5: Center region has dark pixels (0-100)
        center_strength = min(100, (center_dark_ratio / 0.10) * 100) if center_dark_ratio > 0 else 0
        signal_strengths.append(center_strength)
        
        # Signal 6: Contours in head region (0-100, max 20 contours = 100%)
        contour_strength = min(100, (helmet_contours / 20) * 100)
        signal_strengths.append(contour_strength)
        side_strength = min(100, (side_dark_ratio / 0.20) * 100)
        signal_strengths.append(side_strength)
        top_strength = min(100, (top_dark_ratio / 0.18) * 100)
        signal_strengths.append(top_strength)
        shell_strength = min(100, ((1.0 - skin_ratio) / 0.75) * 100)
        signal_strengths.append(shell_strength)
        
        # Calculate confidence as average of all signals
        overall_confidence = np.mean(signal_strengths)
        
        # Decision: need at least 4 out of 6 signals above their thresholds
        # This balances detection accuracy with reducing false positives
        signals_above_threshold = [
            edge_density > threshold_edge,
            dark_ratio > threshold_dark,
            medium_ratio > threshold_medium,
            contrast_mean > threshold_contrast,
            center_dark_ratio > 0.14,
            helmet_contours > 1,
            side_dark_ratio > 0.18,
            top_dark_ratio > 0.16,
            skin_ratio < 0.38,
        ]
        
        signal_count = sum(signals_above_threshold)
        strong_dark_signal = dark_ratio > threshold_dark and medium_ratio > threshold_medium
        strong_shape_signal = (
            center_dark_ratio > 0.14 and
            helmet_contours > 1 and
            (side_dark_ratio > 0.18 or top_dark_ratio > 0.16)
        )

        if detected_face is not None:
            _, fy, _, fh, face_area_ratio = detected_face
            lower_face_visible = (fy + fh) > (h * 0.55)
            shell_like = (
                top_dark_ratio > 0.16 or
                side_dark_ratio > 0.18 or
                (helmet_contours > 1 and medium_ratio > threshold_medium) or
                skin_ratio < 0.34
            )
            if face_area_ratio >= 0.10 and lower_face_visible and not shell_like:
                return False, 8.0

        helmet_detected = (
            signal_count >= 5 and
            strong_dark_signal and
            strong_shape_signal and
            skin_ratio < 0.45 and
            overall_confidence >= HELMET_CONFIDENCE_THRESHOLD
        )
        
        if not helmet_detected:
            overall_confidence = min(overall_confidence * 0.45, HELMET_CONFIDENCE_THRESHOLD - 1)
        
        return helmet_detected, overall_confidence
    except Exception as e:
        print(f"Error in helmet detection: {e}")
        return False, 0.0


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


def format_confidence_label(confidence):
    return f"{max(0.0, min(100.0, float(confidence))):.1f}%"


def draw_labeled_box(frame, x1, y1, x2, y2, label, color, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )
    text_top = max(text_height + 8, y1 - 8)
    label_top = max(0, text_top - text_height - baseline - 6)
    label_bottom = min(frame.shape[0], text_top + baseline - 2)
    label_right = min(frame.shape[1], x1 + text_width + 10)

    cv2.rectangle(frame, (x1, label_top), (label_right, label_bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, text_top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def passes_direct_helmet_threshold(class_name, confidence):
    if is_helmet_class(class_name):
        return confidence >= DIRECT_HELMET_CONFIDENCE_THRESHOLD
    if is_no_helmet_class(class_name):
        return confidence >= DIRECT_NO_HELMET_CONFIDENCE_THRESHOLD
    return True


def overlay_model_boxes(frame, detection_result):
    for box in detection_result.boxes:
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = detection_result.names[int(box.cls[0])]
        confidence = float(box.conf[0]) * 100 if box.conf is not None else 0.0
        if (is_helmet_class(class_name) or is_no_helmet_class(class_name)) and not passes_direct_helmet_threshold(class_name, confidence):
            continue
        label = f"{class_name} {format_confidence_label(confidence)}"
        draw_labeled_box(frame, x1, y1, x2, y2, label, (50, 220, 100))

    return frame


def analyze_helmet_detections(frame, detection_result, model_names):
    classes = [model_names[int(c)] for c in detection_result.boxes.cls]
    explicit_helmet_support = helmet_detection_supported()
    helmet_detected = any(is_helmet_class(class_name) for class_name in classes)
    helmet_count = sum(1 for class_name in classes if is_helmet_class(class_name))
    helmet_confidence = 95.0 if helmet_detected else 0.0
    person_annotations = []
    detected_objects = []

    for box in detection_result.boxes:
        class_name = model_names[int(box.cls[0])]
        confidence = float(box.conf[0]) * 100 if box.conf is not None else 0.0
        detected_objects.append(f"{class_name} ({format_confidence_label(confidence)})")

        if (is_helmet_class(class_name) or is_no_helmet_class(class_name)) and passes_direct_helmet_threshold(class_name, confidence):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 > x1 and y2 > y1:
                person_crop = frame[y1:y2, x1:x2]
                head_assessment = assess_head_coverage(person_crop)
                helmet_label = is_helmet_class(class_name)
                if helmet_label and head_assessment["face_visible"] and not head_assessment["shell_like"]:
                    helmet_label = False
                    confidence = max(confidence, DIRECT_NO_HELMET_CONFIDENCE_THRESHOLD)
                    detected_objects.append("no helmet override (visible face)")
                elif (not helmet_label) and head_assessment["shell_like"] and confidence < DIRECT_NO_HELMET_CONFIDENCE_THRESHOLD + 15:
                    helmet_label = True
                    confidence = max(confidence, DIRECT_HELMET_CONFIDENCE_THRESHOLD)
                    detected_objects.append("helmet override (shell coverage)")

                person_annotations.append({
                    "box": (x1, y1, x2, y2),
                    "helmet": helmet_label,
                    "confidence": confidence,
                })

    if not explicit_helmet_support:
        return {
            "helmet_detected": False,
            "helmet_count": 0,
            "helmet_confidence": 0.0,
            "person_annotations": person_annotations,
            "detected_objects": detected_objects,
            "helmet_status": "Unknown",
        }

    if person_annotations:
        positive_annotations = [annotation for annotation in person_annotations if annotation["helmet"]]
        negative_annotations = [annotation for annotation in person_annotations if not annotation["helmet"]]
        direct_confidences = [annotation["confidence"] for annotation in person_annotations]
        positive_confidence = max((annotation["confidence"] for annotation in positive_annotations), default=0.0)
        negative_confidence = max((annotation["confidence"] for annotation in negative_annotations), default=0.0)
        if direct_confidences:
            helmet_confidence = max(direct_confidences)
        helmet_status = "Unknown"
        helmet_detected = False
        helmet_count = 0
        if positive_confidence >= DIRECT_HELMET_CONFIDENCE_THRESHOLD and positive_confidence > negative_confidence:
            helmet_status = "Yes"
            helmet_detected = True
            helmet_count = len(positive_annotations)
            helmet_confidence = positive_confidence
        elif negative_confidence >= DIRECT_NO_HELMET_CONFIDENCE_THRESHOLD and negative_confidence >= positive_confidence:
            helmet_status = "No"
            helmet_detected = False
            helmet_count = 0
            helmet_confidence = negative_confidence
        return {
            "helmet_detected": helmet_detected,
            "helmet_count": helmet_count,
            "helmet_confidence": helmet_confidence,
            "person_annotations": person_annotations,
            "detected_objects": detected_objects,
            "helmet_status": helmet_status,
        }

    try:
        person_boxes = [
            box for box in detection_result.boxes
            if model_names[int(box.cls[0])] == "person"
        ]

        for box in person_boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]
            is_helmet, confidence = detect_helmet_in_person_crop(person_crop)
            helmet_confidence = max(helmet_confidence, confidence)
            person_annotations.append({
                "box": (x1, y1, x2, y2),
                "helmet": is_helmet,
                "confidence": confidence,
            })
            detected_objects.append(
                f"{'helmet' if is_helmet else 'no helmet'} ({format_confidence_label(confidence)})"
            )

            if not helmet_detected and is_helmet:
                helmet_detected = True
                helmet_count += 1
    except Exception as e:
        print(f"Error checking helmets on persons: {e}")

    return {
        "helmet_detected": helmet_detected,
        "helmet_count": helmet_count,
        "helmet_confidence": helmet_confidence,
        "person_annotations": person_annotations,
        "detected_objects": detected_objects,
        "helmet_status": "Yes" if helmet_detected else "No",
    }


def overlay_person_helmet_annotations(frame, person_annotations):
    for annotation in person_annotations:
        x1, y1, x2, y2 = annotation["box"]
        confidence = annotation["confidence"]
        if annotation["helmet"]:
            label = f"Helmet {format_confidence_label(confidence)}"
            color = (0, 200, 0)
        else:
            label = f"No Helmet {format_confidence_label(confidence)}"
            color = (0, 0, 255)
        draw_labeled_box(frame, x1, y1, x2, y2, label, color, thickness=3)

    return frame


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
    if cached_payload is not None and cached_payload.get("result_version") == IMAGE_RESULT_VERSION:
        return jsonify(cached_payload)

    original_frame = load_image_from_path(path)
    if original_frame is None:
        return jsonify({"error": "Unable to read the uploaded image."}), 400

    frame = resize_for_inference(original_frame)
    results = current_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)
    helmet_analysis = analyze_helmet_detections(frame, results[0], current_model.names)
    result_filename = "result_" + filename
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    # Detect plates using YOLO + minimal OCR (fast)
    annotated_frame = overlay_model_boxes(frame.copy(), results[0])
    plate_numbers = []
    
    custom_plate_model = get_plate_model()
    if custom_plate_model is not None:
        # Use YOLO plate model for detection
        try:
            plate_results = custom_plate_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)
            annotated_frame = overlay_model_boxes(frame.copy(), results[0])
            seen = set()
            
            for box in plate_results[0].boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                plate_crop = frame[y1:y2, x1:x2]
                # Fast OCR on detected plate
                best_text, _ = detect_plate_text_in_crop(plate_crop, fast_mode=True)
                
                if add_plate_result(plate_numbers, seen, best_text):
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    annotate_plate_text(annotated_frame, x1, y1, best_text)
            
            # Fallback: search vehicle regions if no plates detected
            if not plate_numbers:
                helmet_result = results[0]
                vehicle_boxes = [box for box in helmet_result.boxes 
                               if helmet_result.names[int(box.cls[0])] in VEHICLE_CLASSES]
                seen_rects = set()
                
                for box in vehicle_boxes[:MAX_VEHICLE_CROPS]:
                    x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    vehicle_crop = frame[y1:y2, x1:x2]
                    best_text, _ = detect_plate_text_in_crop(vehicle_crop, fast_mode=True)
                    
                    if add_plate_result(plate_numbers, seen, best_text):
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        annotate_plate_text(annotated_frame, x1, y1, best_text)
                        seen_rects.add((x1, y1, x2, y2))
        except Exception as e:
            print(f"Plate detection error: {e}")
            annotated_frame = overlay_model_boxes(frame.copy(), results[0])
    else:
        annotated_frame = overlay_model_boxes(frame.copy(), results[0])

    overlay_person_helmet_annotations(annotated_frame, helmet_analysis["person_annotations"])

    cv2.imwrite(result_path, annotated_frame)

    helmet_detected = helmet_analysis["helmet_detected"]
    helmet_count = helmet_analysis["helmet_count"]
    helmet_confidence = helmet_analysis["helmet_confidence"]
    helmet_status = helmet_analysis.get("helmet_status", "Yes" if helmet_detected else "No")

    payload = {
        "image_path": url_for("static", filename=f"results/{result_filename}"),
        "helmet_status": helmet_status,
        "helmet_detected": helmet_detected,
        "helmet_count": helmet_count,
        "helmet_accuracy": round(helmet_confidence, 2),
        "plate_numbers": plate_numbers,
        "detected_objects": helmet_analysis["detected_objects"],
        "result_version": IMAGE_RESULT_VERSION,
        "detection_summary": (
            f"Helmet detected: {helmet_status} (Accuracy: {helmet_confidence:.1f}%)"
            if helmet_status != "Unknown"
            else "Helmet status: Unknown (dedicated helmet model not available)"
        ),
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
    helmet_confidence = 0.0
    detected_plates = set()

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame = resize_for_inference(frame)
        results = current_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)

        # Check for explicit helmet class
        for r in results:
            for cls in r.boxes.cls:
                class_name = current_model.names[int(cls)]
                if is_helmet_class(class_name):
                    helmet_detected = True
                    helmet_count += 1
                    helmet_confidence = 95.0  # 95% for explicit detection
        
        # If no explicit helmet class, check for helmets on detected persons
        if not helmet_detected:
            try:
                person_boxes = [
                    box for box in results[0].boxes 
                    if current_model.names[int(box.cls[0])] == "person"
                ]
                
                for box in person_boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        person_crop = frame[y1:y2, x1:x2]
                        is_helmet, confidence = detect_helmet_in_person_crop(person_crop)
                        if confidence > helmet_confidence:
                            helmet_confidence = confidence
                        if is_helmet:
                            helmet_detected = True
                            helmet_count += 1
            except Exception as e:
                print(f"Error checking helmets on persons in video: {e}")

        if frame_index % FRAME_SAMPLE_INTERVAL == 0:
            # Quick plate detection on sampled frames
            annotated_frame = results[0].plot()
            custom_plate_model = get_plate_model()
            if custom_plate_model is not None:
                try:
                    plate_results = custom_plate_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)
                    annotated_frame = frame.copy()
                    for box in plate_results[0].boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2 > x1 and y2 > y1:
                            plate_crop = frame[y1:y2, x1:x2]
                            best_text, _ = detect_plate_text_in_crop(plate_crop, fast_mode=True)
                            if best_text:
                                detected_plates.add(best_text)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                annotate_plate_text(annotated_frame, x1, y1, best_text)
                except Exception as e:
                    print(f"Video plate detection error: {e}")
                    annotated_frame = results[0].plot()
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
        "helmet_accuracy": round(helmet_confidence, 2),
        "detection_summary": f"Helmet detected: {helmet_status} (Accuracy: {helmet_confidence:.1f}%)",
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
            
            if frame_index % FRAME_SAMPLE_INTERVAL == 0:
                # Quick plate detection on sampled frames
                frame_display = frame.copy()
                custom_plate_model = get_plate_model()
                if custom_plate_model is not None:
                    try:
                        plate_results = custom_plate_model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)
                        for box in plate_results[0].boxes:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                plate_crop = frame[y1:y2, x1:x2]
                                best_text, _ = detect_plate_text_in_crop(plate_crop, fast_mode=True)
                                if best_text:
                                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                    annotate_plate_text(frame_display, x1, y1, best_text)
                    except Exception as e:
                        print(f"Webcam plate detection error: {e}")
                        frame_display = results[0].plot()
                else:
                    frame_display = results[0].plot()
                frame = frame_display
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
