from __future__ import annotations

import argparse
import base64
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import easyocr
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from app.recognition.plate_rules import best_plate_candidate, clean_plate_text, is_valid_plate


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class SourceItem:
    split: str
    image_path: Path
    label_path: Path
    relative_key: str


class SaveRequest(BaseModel):
    index: int
    text: str


def discover_items(source_dir: Path) -> list[SourceItem]:
    items: list[SourceItem] = []
    for split in ("train", "valid", "test"):
        images_dir = source_dir / split / "images"
        labels_dir = source_dir / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            items.append(SourceItem(
                split=split,
                image_path=image_path,
                label_path=label_path,
                relative_key=f"{split}/images/{image_path.name}",
            ))
    return items


def _bounds_from_row(row: str, width: int, height: int):
    values = [float(value) for value in row.split()[1:]]
    if len(values) == 4:
        center_x, center_y, box_width, box_height = values
        x1 = (center_x - box_width / 2) * width
        x2 = (center_x + box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        y2 = (center_y + box_height / 2) * height
    elif len(values) >= 8 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x1, x2 = min(xs) * width, max(xs) * width
        y1, y2 = min(ys) * height, max(ys) * height
    else:
        return None
    return x1, y1, x2, y2


def crop_plate(item: SourceItem, padding: float = 0.10):
    image = cv2.imread(str(item.image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {item.image_path}")

    height, width = image.shape[:2]
    candidates = []
    for row in item.label_path.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        bounds = _bounds_from_row(row, width, height)
        if bounds is not None:
            x1, y1, x2, y2 = bounds
            candidates.append(((x2 - x1) * (y2 - y1), bounds))
    if not candidates:
        raise RuntimeError(f"No valid YOLO box: {item.label_path}")

    _, (x1, y1, x2, y2) = max(candidates, key=lambda candidate: candidate[0])
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, x2 = max(0, int(x1 - pad_x)), min(width, int(x2 + pad_x))
    y1, y2 = max(0, int(y1 - pad_y)), min(height, int(y2 + pad_y))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop: {item.image_path}")
    return crop


def filename_guess(path: Path) -> str:
    candidate = filename_token(path)
    return candidate if is_valid_full_plate(candidate) else ""


def filename_token(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 2:
        return ""
    candidate = clean_plate_text(parts[1])
    return "" if candidate in {"", "NULL", "NONE"} else candidate


def is_valid_full_plate(text: str) -> bool:
    return is_valid_plate(text)


def _prepare_ocr_image(crop):
    width = crop.shape[1]
    scale = 400 / max(width, 1)
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def ocr_guess(reader: easyocr.Reader, crop) -> tuple[str, float]:
    crop = _prepare_ocr_image(crop)
    results = reader.readtext(
        crop,
        allowlist=ALLOWLIST,
        decoder="beamsearch",
    )
    results.sort(key=lambda result: (
        sum(point[1] for point in result[0]) / len(result[0]),
        sum(point[0] for point in result[0]) / len(result[0]),
    ))
    text = re.sub(r"[^A-Z0-9]", "", "".join(result[1] for result in results).upper())
    confidence = min((float(result[2]) for result in results), default=0.0)
    return text, confidence


def enhanced_crop(crop):
    crop = _prepare_ocr_image(crop)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def threshold_crop(crop):
    crop = _prepare_ocr_image(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def consensus_guess(reader: easyocr.Reader, crop, token: str = "") -> tuple[str, float, int, bool]:
    readings = [
        ocr_guess(reader, crop),
        ocr_guess(reader, enhanced_crop(crop)),
        ocr_guess(reader, threshold_crop(crop)),
    ]
    normalized: list[tuple[str, float]] = []
    for text, confidence in readings:
        candidate = best_plate_candidate(text)
        if candidate is not None:
            # Penalize uncertain character substitutions while still allowing
            # position-aware O/0, I/1, B/8 corrections to reach consensus.
            normalized.append((candidate.text, max(0.0, confidence - candidate.corrections * 0.05)))
        elif text:
            normalized.append((clean_plate_text(text), confidence))

    valid_readings = [(text, confidence) for text, confidence in normalized if is_valid_plate(text)]
    pool = valid_readings or normalized
    if not pool:
        return "", 0.0, 0, False

    token_supported = []
    if len(token) >= 4:
        token_supported = [(text, confidence) for text, confidence in valid_readings if text.endswith(token)]
    if token_supported:
        pool = token_supported

    counts = Counter(text for text, _ in pool)
    suggestion = max(
        counts,
        key=lambda text: (counts[text], max(conf for value, conf in pool if value == text)),
    )
    matching_confidences = [confidence for text, confidence in pool if text == suggestion]
    return (
        suggestion,
        sum(matching_confidences) / len(matching_confidences),
        counts[suggestion],
        bool(token_supported),
    )


def load_annotations(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {key: text for key, text in csv.reader(handle, delimiter="\t")}


def write_annotations(path: Path, annotations: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(sorted(annotations.items()))
    temporary.replace(path)


def load_skipped(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {key: reason for key, reason in csv.reader(handle, delimiter="\t")}


def write_skipped(path: Path, skipped: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(sorted(skipped.items()))
    temporary.replace(path)


def create_app(
    source_dir: Path,
    output: Path,
    gpu: bool = False,
    auto_label_threshold: float | None = None,
) -> FastAPI:
    items = discover_items(source_dir)
    if not items:
        raise RuntimeError(f"No YOLO images found under: {source_dir}")
    valid_keys = {item.relative_key for item in items}
    loaded_annotations = load_annotations(output)
    annotations = {
        key: text for key, text in loaded_annotations.items()
        if key in valid_keys and is_valid_plate(text)
    }
    if annotations != loaded_annotations:
        print(f"Removed {len(loaded_annotations) - len(annotations)} invalid annotation rows")
        write_annotations(output, annotations)
    skipped_path = output.with_name("skipped.tsv")
    skipped = load_skipped(skipped_path)
    reader = easyocr.Reader(["en"], gpu=gpu)
    guess_cache: dict[int, tuple[str, float]] = {}

    if auto_label_threshold is not None:
        accepted_by_filename = 0
        accepted_by_ocr = 0
        pending = 0
        print(f"Auto-labeling {len(items) - len(annotations) - len(skipped)} remaining images...")
        for index, item in enumerate(items):
            if item.relative_key in annotations or item.relative_key in skipped:
                continue
            known_text = filename_guess(item.image_path)
            if known_text:
                annotations[item.relative_key] = known_text
                guess_cache[index] = (known_text, 1.0)
                accepted_by_filename += 1
                continue
            try:
                crop = crop_plate(item)
                token = filename_token(item.image_path)
                suggestion, confidence, votes, token_confirms = consensus_guess(reader, crop, token)
                guess_cache[index] = (suggestion, confidence)
                consensus_accepts = votes >= 2 and confidence >= auto_label_threshold
                filename_accepts = token_confirms and confidence >= 0.60
                if (consensus_accepts or filename_accepts) and is_valid_full_plate(suggestion):
                    annotations[item.relative_key] = suggestion
                    accepted_by_ocr += 1
                else:
                    pending += 1
            except RuntimeError as exc:
                print(f"[SKIP] {item.relative_key}: {exc}")
                skipped[item.relative_key] = str(exc)
            if (index + 1) % 25 == 0:
                write_annotations(output, annotations)
                print(
                    f"Processed {index + 1}/{len(items)} | "
                    f"auto={accepted_by_filename + accepted_by_ocr} | review={pending}"
                )
        write_annotations(output, annotations)
        write_skipped(skipped_path, skipped)
        print(
            f"Auto-label complete: filename={accepted_by_filename}, "
            f"OCR={accepted_by_ocr}, remaining review={pending}"
        )

    app = FastAPI(title="OCR License Plate Labeler")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/item")
    def get_item(index: int = 0):
        if index < 0 or index >= len(items):
            raise HTTPException(status_code=404, detail="Index is outside the dataset")
        item = items[index]
        try:
            crop = crop_plate(item)
        except RuntimeError as exc:
            skipped[item.relative_key] = str(exc)
            write_skipped(skipped_path, skipped)
            return {
                "invalid": True,
                "index": index,
                "total": len(items),
                "detail": str(exc),
            }
        if index not in guess_cache:
            known_text = filename_guess(item.image_path)
            if known_text:
                guess_cache[index] = (known_text, 1.0)
            else:
                suggestion, confidence, _, _ = consensus_guess(
                    reader,
                    crop,
                    filename_token(item.image_path),
                )
                guess_cache[index] = (suggestion, confidence)
        suggestion, confidence = guess_cache[index]
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode crop")
        return {
            "index": index,
            "total": len(items),
            "saved_count": len(annotations),
            "skipped_count": len(skipped),
            "source": item.relative_key,
            "text": annotations.get(item.relative_key, suggestion),
            "confidence": confidence,
            "is_saved": item.relative_key in annotations,
            "image": base64.b64encode(encoded).decode("ascii"),
        }

    @app.get("/api/next-unlabeled")
    def next_unlabeled(after: int = -1):
        for offset in range(1, len(items) + 1):
            index = (after + offset) % len(items)
            key = items[index].relative_key
            if key not in annotations and key not in skipped:
                return {"index": index}
        return {"index": 0, "complete": True}

    @app.post("/api/save")
    def save(request: SaveRequest):
        if request.index < 0 or request.index >= len(items):
            raise HTTPException(status_code=404, detail="Index is outside the dataset")
        candidate = best_plate_candidate(request.text)
        if candidate is None:
            raise HTTPException(status_code=422, detail="Biển số không đúng cấu trúc Việt Nam")
        annotations[items[request.index].relative_key] = candidate.text
        write_annotations(output, annotations)
        return {"ok": True, "saved_count": len(annotations)}

    return app


HTML_PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Gán nhãn biển số</title>
  <style>
    *{box-sizing:border-box} body{margin:0;background:#111827;color:#f9fafb;font-family:Arial,sans-serif}
    main{max-width:920px;margin:36px auto;padding:24px}.card{background:#1f2937;border-radius:18px;padding:28px;box-shadow:0 18px 50px #0006}
    h1{margin:0 0 8px;font-size:28px}.muted{color:#9ca3af}.progress{margin:18px 0;color:#93c5fd}
    .image-wrap{min-height:250px;display:flex;align-items:center;justify-content:center;background:#0b1220;border-radius:12px;padding:22px}
    img{max-width:100%;max-height:430px;image-rendering:auto}.source{margin:14px 0;word-break:break-all}
    input{width:100%;font-size:32px;text-align:center;text-transform:uppercase;letter-spacing:4px;padding:14px;border:2px solid #374151;border-radius:10px;background:#111827;color:white}
    input:focus{outline:none;border-color:#60a5fa}.actions{display:grid;grid-template-columns:1fr 2fr 1fr;gap:12px;margin-top:16px}
    button{border:0;border-radius:10px;padding:14px;font-size:16px;font-weight:700;cursor:pointer;background:#374151;color:white}
    button.primary{background:#2563eb}.saved{color:#34d399}.error{color:#fca5a5;min-height:22px;margin-top:12px}
  </style>
</head>
<body><main><div class="card">
  <h1>Gán nhãn OCR biển số</h1>
  <div class="muted">EasyOCR điền trước — hãy sửa lại cho đúng rồi lưu.</div>
  <div id="progress" class="progress"></div>
  <div class="image-wrap"><img id="plate" alt="Ảnh crop biển số"></div>
  <div id="source" class="source muted"></div>
  <input id="text" maxlength="10" autocomplete="off" spellcheck="false">
  <div class="actions"><button id="previous">← Trước</button><button id="save" class="primary">Lưu và tiếp tục (Enter)</button><button id="next">Sau →</button></div>
  <div id="message" class="error"></div>
</div></main>
<script>
let current=0,total=0;
const text=document.getElementById('text'), message=document.getElementById('message');
async function load(index){
  message.textContent='Đang đọc ảnh…';
  const response=await fetch('/api/item?index='+index);
  if(!response.ok){message.textContent=(await response.json()).detail;return;}
  const data=await response.json(); current=data.index;total=data.total;
  if(data.invalid){
    message.textContent='Đã bỏ qua ảnh lỗi: '+data.detail;
    const next=await fetch('/api/next-unlabeled?after='+current).then(r=>r.json());
    if(!next.complete) load(next.index);
    return;
  }
  document.getElementById('plate').src='data:image/jpeg;base64,'+data.image;
  document.getElementById('source').textContent=data.source;
  document.getElementById('progress').innerHTML=`Ảnh ${current+1}/${total} · Đã lưu ${data.saved_count}/${total} · Bỏ qua ${data.skipped_count} · Tin cậy ${(data.confidence*100).toFixed(1)}% ${data.is_saved?'<span class="saved">· Đã xác nhận</span>':''}`;
  text.value=data.text;message.textContent='';text.focus();text.select();
}
async function save(){
  message.textContent='Đang lưu…';
  const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:current,text:text.value})});
  if(!response.ok){message.textContent=(await response.json()).detail;return;}
  const next=await fetch('/api/next-unlabeled?after='+current).then(r=>r.json());
  if(next.complete){message.textContent='Đã gán nhãn toàn bộ dữ liệu.';return;}
  load(next.index);
}
document.getElementById('save').onclick=save;
document.getElementById('previous').onclick=()=>load((current-1+total)%total);
document.getElementById('next').onclick=()=>load((current+1)%total);
text.addEventListener('keydown',event=>{if(event.key==='Enter')save();});
fetch('/api/next-unlabeled?after=-1').then(r=>r.json()).then(next=>load(next.index));
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Label real license-plate crops with EasyOCR suggestions.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Parent containing train/valid/test folders.")
    parser.add_argument("--output", type=Path, default=Path("data/ocr_annotations/annotations.tsv"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Automatically save filename labels and high-confidence OCR consensus.",
    )
    parser.add_argument("--confidence", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    app = create_app(
        args.source_dir,
        args.output,
        args.gpu,
        args.confidence if args.auto_label else None,
    )
    print(f"Open http://{args.host}:{args.port}")
    print(f"Annotations: {args.output.resolve()}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
