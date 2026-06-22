from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from app.recognition.crnn import ALPHABET


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clean_label(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def find_label(image_path: Path, images_dir: Path, labels_dir: Path) -> Path | None:
    relative = image_path.relative_to(images_dir).with_suffix(".txt")
    candidates = [labels_dir / relative, labels_dir / f"{image_path.stem}.txt"]
    return next((path for path in candidates if path.exists()), None)


def crop_best_detection(image, model: YOLO, padding: float, confidence: float):
    result = model.predict(image, verbose=False, conf=confidence)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best_index = int(result.boxes.conf.argmax().item())
    x1, y1, x2, y2 = result.boxes.xyxy[best_index].detach().cpu().tolist()
    h, w = image.shape[:2]
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x2, y2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
    return image[y1:y2, x1:x2]


def split_groups(labels: list[str], seed: int, train_ratio: float, val_ratio: float):
    unique_labels = sorted(set(labels))
    random.Random(seed).shuffle(unique_labels)
    count = len(unique_labels)
    train_end = int(count * train_ratio)
    val_end = train_end + int(count * val_ratio)
    assignment = {}
    for label in unique_labels[:train_end]:
        assignment[label] = "train"
    for label in unique_labels[train_end:val_end]:
        assignment[label] = "val"
    for label in unique_labels[val_end:]:
        assignment[label] = "test"
    return assignment


def split_two_line_crop(crop, label: str):
    if len(label) not in (8, 9):
        return [(crop, label)]
    h = crop.shape[0]
    overlap = max(1, int(h * 0.04))
    middle = h // 2
    return [
        (crop[:middle + overlap], label[:4]),
        (crop[middle - overlap:], label[4:]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leakage-safe OCR dataset from vehicle images.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ocr"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--already-cropped", action="store_true")
    parser.add_argument("--two-line", action="store_true", help="Split motorbike plates into two OCR samples.")
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Ratios must satisfy train > 0, val >= 0 and train + val < 1")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_images = args.output_dir / "images"
    output_images.mkdir(exist_ok=True)

    model = None if args.already_cropped else YOLO(str(args.detector_model))
    records = []
    failures = []
    image_paths = sorted(path for path in args.images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)

    for index, image_path in enumerate(image_paths):
        label_path = find_label(image_path, args.images_dir, args.labels_dir)
        if label_path is None:
            failures.append((str(image_path), "missing_label"))
            continue
        label = clean_label(label_path.read_text(encoding="utf-8").strip())
        if not label or any(char not in ALPHABET for char in label):
            failures.append((str(image_path), "invalid_label"))
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            failures.append((str(image_path), "invalid_image"))
            continue
        crop = image if args.already_cropped else crop_best_detection(image, model, args.padding, args.confidence)
        if crop is None or crop.size == 0:
            failures.append((str(image_path), "no_detection"))
            continue

        samples = split_two_line_crop(crop, label) if args.two_line else [(crop, label)]
        for line_index, (sample, sample_label) in enumerate(samples):
            filename = f"{index:07d}_{line_index}.jpg"
            cv2.imwrite(str(output_images / filename), sample)
            records.append((f"images/{filename}", sample_label, label))

    assignments = split_groups([record[2] for record in records], args.seed, args.train_ratio, args.val_ratio)
    manifests = defaultdict(list)
    for relative_path, sample_label, group_label in records:
        manifests[assignments[group_label]].append((relative_path, sample_label))

    for split in ("train", "val", "test"):
        with (args.output_dir / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerows(manifests[split])
    with (args.output_dir / "failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "reason"])
        writer.writerows(failures)

    print(f"Prepared {len(records)} OCR samples from {len(image_paths)} images")
    print("Split sizes:", {split: len(manifests[split]) for split in ("train", "val", "test")})
    print(f"Failures: {len(failures)} (see {args.output_dir / 'failures.csv'})")


if __name__ == "__main__":
    main()

