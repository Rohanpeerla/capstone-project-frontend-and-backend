from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "car number plate dataset"
OUTPUT_DIR = BASE_DIR / "yolo_plate_dataset"
RANDOM_SEED = 42
VAL_RATIO = 0.2
CLASS_NAME = "number_plate"


def parse_box(obj, width: int, height: int):
    bndbox = obj.find("bndbox")
    if bndbox is None:
        return None

    xmin = float(bndbox.findtext("xmin", "0"))
    ymin = float(bndbox.findtext("ymin", "0"))
    xmax = float(bndbox.findtext("xmax", "0"))
    ymax = float(bndbox.findtext("ymax", "0"))

    if xmax <= xmin or ymax <= ymin or width <= 0 or height <= 0:
        return None

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def collect_samples():
    samples = []

    xml_groups = [
        (SOURCE_DIR / "Annotations" / "Annotations", SOURCE_DIR / "Indian_Number_Plates" / "Sample_Images"),
        (SOURCE_DIR / "number_plate_annos_ocr" / "number_plate_annos_ocr", SOURCE_DIR / "number_plate_images_ocr" / "number_plate_images_ocr"),
    ]

    for xml_dir, image_dir in xml_groups:
        if not xml_dir.exists() or not image_dir.exists():
            continue

        for xml_path in xml_dir.glob("*.xml"):
            root = ET.parse(xml_path).getroot()
            filename = root.findtext("filename")
            width = int(root.findtext("size/width", "0"))
            height = int(root.findtext("size/height", "0"))
            if not filename or width <= 0 or height <= 0:
                continue

            image_path = image_dir / filename
            if not image_path.exists():
                continue

            labels = []
            for obj in root.findall("object"):
                if obj.findtext("name") != CLASS_NAME:
                    continue
                label_line = parse_box(obj, width, height)
                if label_line:
                    labels.append(label_line)

            if labels:
                samples.append((image_path, labels))

    return samples


def ensure_output_dirs():
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    data_yaml = OUTPUT_DIR / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_DIR.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "nc: 1",
                "names: [number_plate]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    samples = collect_samples()
    if not samples:
        raise FileNotFoundError("No usable number plate samples found in the car number plate dataset.")

    ensure_output_dirs()
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    split_index = max(1, int(len(samples) * (1 - VAL_RATIO)))
    train_samples = samples[:split_index]
    val_samples = samples[split_index:] or samples[-1:]

    for split, split_samples in (("train", train_samples), ("val", val_samples)):
        for image_path, labels in split_samples:
            target_image = OUTPUT_DIR / "images" / split / image_path.name
            target_label = OUTPUT_DIR / "labels" / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")

    write_data_yaml()
    print(f"Prepared {len(train_samples)} train and {len(val_samples)} val images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
