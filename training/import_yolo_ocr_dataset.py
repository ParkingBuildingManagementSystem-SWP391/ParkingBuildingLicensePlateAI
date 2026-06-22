from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path

import cv2

from app.recognition.crnn import ALPHABET


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def plate_from_filename(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 2:
        return ""
    return re.sub(r"[^A-Z0-9]", "", parts[1].upper())


def is_plausible_full_plate(text: str) -> bool:
    """Reject partial OCR-like filenames such as `17054` or `NULL`."""
    return re.fullmatch(r"\d{2}[A-Z]{1,2}\d{4,6}", text) is not None


def crop_from_yolo_label(image, label_path: Path, padding: float = 0.05):
    rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    values = [float(value) for value in rows[0].split()[1:]]
    h, w = image.shape[:2]
    if len(values) == 4:
        center_x, center_y, box_w, box_h = values
        x1, x2 = (center_x - box_w / 2) * w, (center_x + box_w / 2) * w
        y1, y2 = (center_y - box_h / 2) * h, (center_y + box_h / 2) * h
    elif len(values) >= 8 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x1, x2 = min(xs) * w, max(xs) * w
        y1, y2 = min(ys) * h, max(ys) * h
    else:
        return None

    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, x2 = max(0, int(x1 - pad_x)), min(w, int(x2 + pad_x))
    y1, y2 = max(0, int(y1 - pad_y)), min(h, int(y2 + pad_y))
    return image[y1:y2, x1:x2]


def split_plate(crop, label: str, top_length: int = 3):
    h, w = crop.shape[:2]
    if w / max(h, 1) >= 2.0 or len(label) <= top_length:
        return [(crop, label)]
    overlap = max(1, int(h * 0.04))
    middle = h // 2
    return [
        (crop[:middle + overlap], label[:top_length]),
        (crop[middle - overlap:], label[top_length:]),
    ]


def group_assignment(labels: list[str], seed: int):
    unique = sorted(set(labels))
    random.Random(seed).shuffle(unique)
    train_end = int(len(unique) * 0.8)
    val_end = train_end + int(len(unique) * 0.1)
    return {
        label: ("train" if index < train_end else "val" if index < val_end else "test")
        for index, label in enumerate(unique)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a YOLO plate dataset whose filenames contain OCR labels.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Parent containing train/valid/test folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ocr_external"))
    parser.add_argument("--top-length", type=int, default=3, help="Characters on the top row of square car plates.")
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    images_output = args.output_dir / "images"
    images_output.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []
    index = 0
    for source_split in ("train", "valid", "test"):
        images_dir = args.source_dir / source_split / "images"
        labels_dir = args.source_dir / source_split / "labels"
        for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            plate = plate_from_filename(image_path)
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not plate or any(char not in ALPHABET for char in plate) or not is_plausible_full_plate(plate):
                failures.append((str(image_path), "incomplete_or_invalid_plate_in_filename"))
                continue
            if not label_path.exists():
                failures.append((str(image_path), "missing_yolo_label"))
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                failures.append((str(image_path), "invalid_image"))
                continue
            crop = crop_from_yolo_label(image, label_path, args.padding)
            if crop is None or crop.size == 0:
                failures.append((str(image_path), "invalid_yolo_label"))
                continue
            samples = split_plate(crop, plate, args.top_length)
            for line_index, (sample, text) in enumerate(samples):
                relative = f"images/{index:07d}_{line_index}.jpg"
                cv2.imwrite(str(args.output_dir / relative), sample)
                records.append((relative, text, plate))
            index += 1

    assignments = group_assignment([plate for _, _, plate in records], args.seed)
    split_records = {"train": [], "val": [], "test": []}
    for relative, text, plate in records:
        split_records[assignments[plate]].append((relative, text))
    for split, rows in split_records.items():
        with (args.output_dir / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)
    with (args.output_dir / "failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "reason"])
        writer.writerows(failures)

    print(f"Imported source images: {index}")
    print(f"OCR line samples: {len(records)}")
    print("Split sizes:", {split: len(rows) for split, rows in split_records.items()})
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
