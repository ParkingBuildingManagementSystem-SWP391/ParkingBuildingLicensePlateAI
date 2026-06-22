from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROVINCE_CODES = [
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "36", "37", "38", "40", "41",
    "43", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73",
    "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "88",
    "89", "90", "92", "93", "94", "95", "97", "98", "99",
]
SERIES = "ABCDEFGHKLMNPSTUVXYZ"


def random_label(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.35:  # Top row of a two-line motorbike plate.
        if rng.random() < 0.15:
            return rng.choice(PROVINCE_CODES) + rng.choice("AM") + rng.choice(SERIES)
        return rng.choice(PROVINCE_CODES) + rng.choice(SERIES) + str(rng.randint(1, 9))
    if kind < 0.75:  # Numeric bottom row.
        length = 5 if rng.random() < 0.8 else 4
        return "".join(str(rng.randint(0, 9)) for _ in range(length))
    # One-line car plate.
    return rng.choice(PROVINCE_CODES) + rng.choice(SERIES) + "".join(str(rng.randint(0, 9)) for _ in range(5))


def find_font(font_path: Path | None) -> str:
    candidates = [
        font_path,
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No suitable TrueType font found; pass --font explicitly")


def render(label: str, font_path: str, rng: random.Random) -> np.ndarray:
    width, height = 220, 64
    background, foreground = rng.choice([
        ((245, 245, 235), (15, 15, 15)),
        ((242, 207, 35), (20, 20, 20)),
        ((35, 80, 145), (245, 245, 245)),
    ])
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    font_size = rng.randint(39, 49)
    font = ImageFont.truetype(font_path, font_size)
    box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    text_w, text_h = box[2] - box[0], box[3] - box[1]
    x = max(3, (width - text_w) // 2 + rng.randint(-4, 4))
    y = max(-5, (height - text_h) // 2 - box[1] + rng.randint(-3, 3))
    draw.text((x, y), label, font=font, fill=foreground, stroke_width=rng.choice((0, 0, 1)), stroke_fill=foreground)

    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # Mild camera-like perspective distortion.
    jitter = 5
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst = src + np.float32([[rng.randint(-jitter, jitter), rng.randint(-jitter, jitter)] for _ in range(4)])
    matrix = cv2.getPerspectiveTransform(src, dst)
    array = cv2.warpPerspective(array, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.35:
        array = cv2.GaussianBlur(array, (3, 3), rng.uniform(0.2, 1.2))
    if rng.random() < 0.5:
        alpha, beta = rng.uniform(0.7, 1.25), rng.randint(-20, 20)
        array = cv2.convertScaleAbs(array, alpha=alpha, beta=beta)
    if rng.random() < 0.25:
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(2, 8), array.shape)
        array = np.clip(array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return array


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic single-line Vietnamese plate OCR samples.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ocr_synthetic"))
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    font_path = find_font(args.font)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "train.tsv"
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest} already exists; use --overwrite")

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for index in range(args.count):
            label = random_label(rng)
            filename = f"images/{index:07d}.jpg"
            cv2.imwrite(str(args.output_dir / filename), render(label, font_path, rng))
            writer.writerow([filename, label])
    print(f"Generated {args.count} samples in {args.output_dir} using {font_path}")


if __name__ == "__main__":
    main()

