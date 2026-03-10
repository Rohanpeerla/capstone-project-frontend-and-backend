from pathlib import Path
import shutil

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "yolo_plate_dataset" / "data.yaml"
BASE_MODEL = BASE_DIR / "yolov8n.pt"
OUTPUT_DIR = BASE_DIR / "training_runs"
MODEL_OUTPUT = BASE_DIR / "plate_model.pt"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}. Run prepare_plate_dataset.py first."
        )

    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")

    model = YOLO(str(BASE_MODEL))
    results = model.train(
        data=str(DATA_YAML),
        epochs=25,
        imgsz=640,
        batch=8,
        workers=0,
        project=str(OUTPUT_DIR),
        name="plate_model",
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Trained model not found: {best_weights}")

    shutil.copy2(best_weights, MODEL_OUTPUT)
    print(f"Saved trained model to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
