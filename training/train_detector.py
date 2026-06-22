from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the license-plate YOLO detector.")
    parser.add_argument("--data", type=Path, default=Path("data/detection/data.yaml"))
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str((Path.cwd() / "runs" / "detect").resolve()),
        name="plate_detector",
        patience=20,
    )


if __name__ == "__main__":
    main()
