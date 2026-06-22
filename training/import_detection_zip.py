from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_yolo_label(text: str) -> bool:
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            return False
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            return False
        if class_id != 0 or any(value < 0 or value > 1 for value in coords):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and split a flat YOLO detection ZIP dataset.")
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/detection"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Ratios must satisfy train > 0, val > 0 and train + val < 1")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)

    with zipfile.ZipFile(args.zip_path) as archive:
        entries = {PurePosixPath(info.filename).name: info for info in archive.infolist() if not info.is_dir()}
        image_names = sorted(name for name in entries if Path(name).suffix.lower() in IMAGE_EXTENSIONS)
        pairs = []
        failures = []
        for image_name in image_names:
            label_name = f"{Path(image_name).stem}.txt"
            if label_name not in entries:
                failures.append(f"{image_name}: missing label")
                continue
            label_text = archive.read(entries[label_name]).decode("utf-8-sig")
            if not validate_yolo_label(label_text):
                failures.append(f"{image_name}: invalid YOLO label")
                continue
            pairs.append((image_name, label_name, label_text))

        random.Random(args.seed).shuffle(pairs)
        train_end = int(len(pairs) * args.train_ratio)
        val_end = train_end + int(len(pairs) * args.val_ratio)
        split_pairs = {
            "train": pairs[:train_end],
            "val": pairs[train_end:val_end],
            "test": pairs[val_end:],
        }

        for split, items in split_pairs.items():
            image_dir = args.output_dir / "images" / split
            label_dir = args.output_dir / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for image_name, label_name, label_text in items:
                (image_dir / image_name).write_bytes(archive.read(entries[image_name]))
                (label_dir / label_name).write_text(label_text, encoding="utf-8")

    root = args.output_dir.resolve().as_posix()
    (args.output_dir / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: plate\n",
        encoding="utf-8",
    )
    (args.output_dir / "failures.txt").write_text("\n".join(failures), encoding="utf-8")
    print(f"Imported valid pairs: {len(pairs)}")
    print("Split sizes:", {split: len(items) for split, items in split_pairs.items()})
    print(f"Failures: {len(failures)}")
    print(f"YOLO config: {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

