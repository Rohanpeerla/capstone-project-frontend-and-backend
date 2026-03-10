from pathlib import Path
import shutil
import yaml

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIRS = [
    BASE_DIR / "traffic dataset",
    BASE_DIR / "dataset",
    Path(r"C:\Users\sairo\Downloads\traffic dataset"),
]
DEFAULT_DATA_YAML = BASE_DIR / "data.yaml"
DEFAULT_BASE_MODEL = BASE_DIR / "yolov8n.pt"
OUTPUT_DIR = BASE_DIR / "training_runs"
MODEL_OUTPUT = BASE_DIR / "traffic_model.pt"


def find_dataset_yaml() -> Path:
    if DEFAULT_DATA_YAML.exists():
        return DEFAULT_DATA_YAML

    for dataset_dir in DEFAULT_DATASET_DIRS:
        data_yaml = dataset_dir / "data.yaml"
        if data_yaml.exists():
            return data_yaml

    raise FileNotFoundError(
        "No data.yaml found. Put your YOLO dataset in the workspace or create data.yaml in this folder."
    )


def resolve_dataset_paths(data_yaml: Path) -> tuple[Path, Path, Path]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    dataset_root = Path(data.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    train_path = (dataset_root / data["train"]).resolve()
    val_path = (dataset_root / data["val"]).resolve()
    return dataset_root, train_path, val_path


def validate_yolo_dataset(data_yaml: Path):
    dataset_root, train_path, val_path = resolve_dataset_paths(data_yaml)

    missing_paths = [str(path) for path in (train_path, val_path) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Dataset paths from data.yaml do not exist:\n"
            + "\n".join(missing_paths)
        )

    labels_root = dataset_root / "labels"
    if not labels_root.exists():
        raise FileNotFoundError(
            f"Expected YOLO labels folder at {labels_root}, but it does not exist."
        )

    train_labels = labels_root / "train"
    val_labels = labels_root / "val"
    if not train_labels.exists() or not val_labels.exists():
        raise FileNotFoundError(
            "Expected YOLO labels/train and labels/val folders, but they were not found."
        )

    if not any(train_labels.rglob("*.txt")):
        raise FileNotFoundError(f"No label .txt files found in {train_labels}")
    if not any(val_labels.rglob("*.txt")):
        raise FileNotFoundError(f"No label .txt files found in {val_labels}")


def main():
    data_yaml = find_dataset_yaml()
    validate_yolo_dataset(data_yaml)

    base_model = DEFAULT_BASE_MODEL
    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")

    model = YOLO(str(base_model))
    results = model.train(
        data=str(data_yaml),
        epochs=25,
        imgsz=640,
        batch=8,
        workers=0,
        project=str(OUTPUT_DIR),
        name="traffic_model",
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Trained model not found: {best_weights}")

    shutil.copy2(best_weights, MODEL_OUTPUT)
    print(f"Saved trained model to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
