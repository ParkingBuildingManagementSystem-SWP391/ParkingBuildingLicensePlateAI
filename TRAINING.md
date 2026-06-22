# Huấn luyện OCR biển số Việt Nam

Project dùng hai model độc lập:

- `models/best.pt`: YOLO phát hiện vùng biển số.
- `models/ocr_crnn.pt`: CRNN–CTC đọc ký tự trên từng dòng biển số.

## Huấn luyện YOLO detector từ ZIP

Dataset chỉ có ảnh và nhãn tọa độ YOLO được dùng để train `best.pt`, không dùng để train OCR:

```powershell
python -m training.import_detection_zip `
  --zip "C:\Users\Admin\Downloads\yolo_plate_dataset.zip" `
  --output-dir data\detection

python -m training.train_detector `
  --data data\detection\data.yaml `
  --model models\yolov8n.pt `
  --epochs 100 `
  --batch-size 8 `
  --device 0
```

Checkpoint tốt nhất nằm tại `runs/detect/plate_detector/weights/best.pt`. Chỉ thay `models/best.pt` sau khi kết quả validation tốt hơn model hiện tại.

Nếu chưa có `ocr_crnn.pt`, API tiếp tục dùng EasyOCR. Vì vậy việc chuẩn bị hoặc train lỗi không làm gián đoạn hệ thống hiện tại.

## 1. Tạo dataset thật

Mỗi ảnh trong thư mục nhãn cần có một file `.txt` cùng tên, nội dung là biển số đúng, ví dụ `59X312345`.

```powershell
python -m training.prepare_ocr_dataset `
  --images-dir "C:\duong-dan\anh" `
  --labels-dir "C:\duong-dan\label" `
  --detector-model models\best.pt `
  --output-dir data\ocr `
  --two-line `
  --overwrite
```

`--two-line` dành cho tập xe máy. Công cụ dùng YOLO để crop, tách hai dòng và chia train/validation/test theo biển số, không theo ảnh. Các ảnh lỗi nằm trong `data/ocr/failures.csv`.

Nếu ảnh nguồn đã là crop biển số, thêm `--already-cropped` và bỏ qua detector.

## 2. Sinh dữ liệu tổng hợp để pretrain

Không nên train CRNN từ đầu chỉ với vài trăm ảnh thật. Tạo tối thiểu 20.000–100.000 dòng tổng hợp:

```powershell
python -m training.generate_synthetic --count 50000 --overwrite
```

Dữ liệu tổng hợp bao gồm dòng trên xe máy, dòng số phía dưới và biển ô tô một dòng, với biến đổi phối cảnh, ánh sáng, blur, nhiễu và màu nền.

Nếu có dataset YOLO `train/valid/test` với đáp án biển số nằm trong tên file, có thể nhập trực tiếp thay cho synthetic:

```powershell
python -m training.import_yolo_ocr_dataset `
  --source-dir "C:\Users\Admin\Downloads" `
  --output-dir data\ocr_external `
  --top-length 3 `
  --overwrite
```

## 3. Pretrain bằng synthetic, fine-tune bằng ảnh thật

Pretrain (nên dùng GPU NVIDIA):

```powershell
python -m training.train_crnn `
  --data-dir data\ocr `
  --extra-train-manifest data\ocr_synthetic\train.tsv `
  --output models\ocr_crnn_pretrain.pt `
  --epochs 30 `
  --batch-size 128
```

Fine-tune chỉ trên dữ liệu thật với learning rate thấp:

```powershell
python -m training.train_crnn `
  --data-dir data\ocr `
  --resume models\ocr_crnn_pretrain.pt `
  --output models\ocr_crnn.pt `
  --epochs 50 `
  --batch-size 32 `
  --learning-rate 0.0001
```

Khởi động lại FastAPI sau khi checkpoint được tạo. Service sẽ tự phát hiện và sử dụng model mới.

## 4. Điều kiện đưa vào sử dụng

Không đổi tên một checkpoint thử nghiệm thành `ocr_crnn.pt` nếu chưa đạt tối thiểu:

- Character accuracy trên validation: 98%.
- Exact accuracy trên validation độc lập: 90%.
- Kết quả end-to-end trên `batch_test.py` cao hơn EasyOCR hiện tại.

Tập 116 ảnh hiện tại đã tạo được 221 mẫu dòng: 175 train, 22 validation và 24 test; 5 ảnh chưa sử dụng được. Quy mô này phù hợp để kiểm tra pipeline nhưng chưa đủ để train từ đầu.
