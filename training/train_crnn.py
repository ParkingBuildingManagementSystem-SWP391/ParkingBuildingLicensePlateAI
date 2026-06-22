from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from app.recognition.crnn import ALPHABET, CRNN, greedy_decode, normalize_plate_image


class PlateDataset(Dataset):
    def __init__(self, manifest: Path, augment: bool = False):
        self.root = manifest.parent
        self.augment = augment
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            self.samples = [(path, label) for path, label in csv.reader(handle, delimiter="\t")]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        # Some JPEG crops can be decoded into a non-contiguous array that makes
        # particular OpenCV builds fail inside GaussianBlur. Normalize the
        # memory layout first and skip only the failing augmentation operation.
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if random.random() < 0.5:
            alpha = random.uniform(0.65, 1.35)
            beta = random.uniform(-25, 25)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        if random.random() < 0.25:
            kernel = random.choice((3, 5))
            try:
                image = cv2.GaussianBlur(image, (kernel, kernel), 0)
            except cv2.error:
                # Augmentation is optional; the original crop remains valid.
                pass
        if random.random() < 0.20:
            quality = random.randint(35, 85)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return image

    def __getitem__(self, index):
        relative_path, label = self.samples[index]
        image = cv2.imread(str(self.root / relative_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {self.root / relative_path}")
        if self.augment:
            image = self._augment(image)
        return normalize_plate_image(image), label


def encode_labels(labels: list[str], char_to_index: dict[str, int]):
    encoded = [char_to_index[char] for label in labels for char in label]
    lengths = [len(label) for label in labels]
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def evaluate(model, loader, device):
    model.eval()
    exact = total = edits = characters = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = greedy_decode(model(images.to(device)))
            for prediction, label in zip(predictions, labels):
                exact += int(prediction == label)
                total += 1
                edits += edit_distance(prediction, label)
                characters += max(len(label), len(prediction), 1)
    return {
        "exact_accuracy": exact / max(total, 1),
        "character_accuracy": 1.0 - edits / max(characters, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Vietnamese plate CRNN recognizer.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ocr"))
    parser.add_argument("--output", type=Path, default=Path("models/ocr_crnn.pt"))
    parser.add_argument("--extra-train-manifest", type=Path, action="append", default=[])
    parser.add_argument("--resume", type=Path, help="Load model weights from an existing checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    train_datasets = [PlateDataset(args.data_dir / "train.tsv", augment=True)]
    train_datasets.extend(PlateDataset(path, augment=True) for path in args.extra_train_manifest)
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = PlateDataset(args.data_dir / "val.tsv")
    if not train_dataset or not val_dataset:
        raise RuntimeError("Train and validation manifests must both contain samples")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = CRNN().to(device)
    start_epoch = 1
    best_score = (-1.0, -1.0)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_score = (
            float(checkpoint.get("val_exact_accuracy", -1.0)),
            float(checkpoint.get("val_character_accuracy", -1.0)),
        )
        print(f"Resumed weights from {args.resume} at epoch {start_epoch}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
    if args.resume:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    char_to_index = {char: index + 1 for index, char in enumerate(ALPHABET)}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    if len(train_dataset) < 5000:
        print("[WARNING] Fewer than 5,000 training samples; this is only suitable for a pipeline smoke test.")

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"set --epochs to a value of at least {start_epoch}."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            targets, target_lengths = encode_labels(list(labels), char_to_index)
            targets, target_lengths = targets.to(device), target_lengths.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["exact_accuracy"])
        print(
            f"Epoch {epoch:03d} | loss={running_loss / max(len(train_loader), 1):.4f} "
            f"| val_exact={metrics['exact_accuracy']:.4f} "
            f"| val_char={metrics['character_accuracy']:.4f}"
        )
        # Prefer full-plate accuracy, then character accuracy while exact accuracy is tied.
        score = (metrics["exact_accuracy"], metrics["character_accuracy"])
        if score > best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "alphabet": ALPHABET,
                "image_height": 32,
                "image_width": 160,
                "epoch": epoch,
                "val_exact_accuracy": metrics["exact_accuracy"],
                "val_character_accuracy": metrics["character_accuracy"],
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, args.output)
            print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
