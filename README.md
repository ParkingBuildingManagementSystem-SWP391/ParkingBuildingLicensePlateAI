---
title: Parking License Plate AI
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🚘 Parking Building License Plate AI

### Phát hiện nhanh, đọc chính xác, tích hợp đơn giản

Dịch vụ ALPR nhận diện biển số xe Việt Nam, xây dựng bằng FastAPI, YOLOv8 và EasyOCR.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=for-the-badge&logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/ParkingBuildingManagementSystem-SWP391/ParkingBuildingLicensePlateAI)
[![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey?style=for-the-badge)](#-license--contact)

[Giới thiệu](#-about-the-project) •
[Tính năng](#-key-features) •
[Cài đặt](#-getting-started) •
[API](#-usage) •
[Tài liệu luồng](LicensePlateRecognitionFlow.md)

</div>

---

## 🚀 About The Project

**Parking Building License Plate AI** là microservice AI độc lập dành cho hệ thống quản lý bãi đỗ xe. Backend .NET gửi trực tiếp file ảnh phương tiện đến FastAPI; dịch vụ phát hiện vùng biển số, nhận dạng ký tự và trả về chuỗi biển số Việt Nam đã được chuẩn hóa.

Luồng nhận diện không phụ thuộc vào Cloudinary hoặc một dịch vụ lưu trữ ảnh trung gian:

```text
Frontend → Backend .NET → FastAPI → YOLOv8 → Image Processing
                                      → EasyOCR → Plate Normalization → Backend
```

> **Lưu ý kiến trúc:** AI chỉ chịu trách nhiệm nhận diện. Việc xác nhận kết quả, lưu ảnh và lưu phiên gửi xe thuộc trách nhiệm của Backend.

### Quy trình xử lý

1. FastAPI nhận file ảnh qua `multipart/form-data`.
2. OpenCV giải mã và điều chỉnh kích thước ảnh đầu vào.
3. YOLOv8 phát hiện vùng biển số có độ tin cậy cao nhất.
4. Vùng biển số được cắt, sửa nghiêng, tăng tương phản và khử nhiễu.
5. EasyOCR đọc ký tự với allowlist `0-9` và `A-Z`.
6. Bộ quy tắc hậu xử lý chuẩn hóa kết quả theo cấu trúc biển số Việt Nam.
7. API trả trường `license_plate` cho Backend.

---

## 🛠️ Built With

| Công nghệ | Vai trò |
|---|---|
| **Python 3.11** | Ngôn ngữ phát triển chính |
| **FastAPI** | Xây dựng REST API bất đồng bộ |
| **Uvicorn** | ASGI server để vận hành ứng dụng |
| **Ultralytics YOLOv8** | Phát hiện vị trí biển số trong ảnh |
| **EasyOCR** | Nhận dạng ký tự trên vùng biển số |
| **OpenCV** | Giải mã, cắt, sửa nghiêng và tiền xử lý ảnh |
| **NumPy** | Biểu diễn và thao tác dữ liệu ảnh |
| **PyTorch** | Runtime cho các mô hình AI |
| **Jinja2** | Render Web Dashboard kiểm thử |
| **Docker** | Đóng gói và triển khai nhất quán |

---

## ✨ Key Features

- 🔍 **Phát hiện biển số tự động** bằng YOLOv8.
- 🔤 **Nhận dạng ký tự** với EasyOCR và tập ký tự được giới hạn phù hợp.
- 🇻🇳 **Chuẩn hóa biển số Việt Nam**, bao gồm biển một dòng và hai dòng.
- 🧹 **Tiền xử lý ảnh chuyên biệt**: sửa nghiêng, CLAHE, Otsu threshold và morphology.
- ⚡ **Fast mode** giúp giảm số lần OCR và thời gian phản hồi.
- 🧠 **Lựa chọn ứng viên thông minh** dựa trên định dạng biển số và confidence.
- 🔌 **API tối ưu cho Backend .NET**, gửi trực tiếp file ảnh và nhận JSON gọn nhẹ.
- 🖼️ **Dashboard trực quan** hiển thị biển số, bounding box và ảnh crop.
- 🔥 **Model warm-up** giảm độ trễ của request thực tế đầu tiên.
- 🐳 **Docker-ready** và hỗ trợ triển khai trên Hugging Face Spaces.
- ✅ **Unit test** cho các quy tắc tiền xử lý và chuẩn hóa biển số.

---

## 🏁 Getting Started

### Prerequisites

Đảm bảo máy đã cài:

- [Python 3.11+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/) — tùy chọn

> **Khuyến nghị:** Sử dụng virtual environment. Nếu bật `OCR_GPU = True`, máy cần có GPU NVIDIA và môi trường CUDA/PyTorch tương thích.

### 1. Clone repository

```bash
git clone https://github.com/ParkingBuildingManagementSystem-SWP391/ParkingBuildingLicensePlateAI.git
cd ParkingBuildingLicensePlateAI
```

### 2. Tạo virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Chuẩn bị model

Đặt model YOLO đã huấn luyện tại:

```text
models/best.pt
```

Nếu file này không tồn tại, ứng dụng sẽ sử dụng model fallback:

```text
keremberke/yolov8n-license-plate-detector
```

> Lần chạy đầu có thể lâu hơn do tải model fallback và khởi tạo YOLO/EasyOCR. Khi triển khai production, nên cung cấp sẵn `models/best.pt`.

### 5. Chạy ứng dụng

```bash
python app/main.py
```

Hoặc chạy trực tiếp với Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Sau khi khởi động:

| Chức năng | URL |
|---|---|
| Web Dashboard | `http://127.0.0.1:8000/` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| Health check | `http://127.0.0.1:8000/health` |

### Chạy bằng Docker

```bash
docker build -t parking-license-plate-ai .
docker run --rm -p 7860:7860 parking-license-plate-ai
```

Truy cập `http://127.0.0.1:7860` sau khi container khởi động.

---

## 📖 Usage

### API dành cho Backend .NET

```http
POST /predict-file-fast
Content-Type: multipart/form-data
```

| Tham số | Kiểu | Bắt buộc | Mô tả |
|---|---|:---:|---|
| `file` | Image file | ✅ | Ảnh phương tiện cần nhận diện |

Ví dụ với cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict-file-fast" \
  -H "accept: application/json" \
  -F "file=@vehicle.jpg"
```

Response thành công:

```json
{
  "status": "success",
  "license_plate": "59X312345"
}
```

Response khi ảnh không có biển số:

```json
{
  "status": "error",
  "message": "Không phát hiện thấy biển số xe"
}
```

### API dành cho Web Dashboard

```http
POST /predict-file
Content-Type: multipart/form-data
```

Endpoint này trả thêm `annotated_image` và `crop_image` dưới dạng Base64 để hiển thị trực quan. Backend production nên sử dụng `/predict-file-fast` để tránh response lớn.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "service": "Parking Building License Plate AI"
}
```

---

## 📁 Project Structure

```text
ParkingBuildingLicensePlateAI/
├── app/
│   ├── api/
│   │   └── predict.py          # API upload và nhận diện ảnh
│   ├── core/
│   │   └── config.py           # Cấu hình model và OCR
│   ├── recognition/
│   │   └── plate_rules.py      # Quy tắc biển số Việt Nam
│   ├── services/
│   │   └── detector.py         # Pipeline YOLO + EasyOCR
│   ├── templates/
│   │   └── index.html          # Web Dashboard
│   ├── utils/
│   │   ├── helpers.py          # Hậu xử lý kết quả OCR
│   │   └── plate_image.py      # Tiền xử lý vùng biển số
│   └── main.py                 # FastAPI entry point
├── models/
│   └── best.pt                 # Model YOLO tùy chỉnh
├── tests/
│   └── test_recognition.py     # Unit test pipeline nhận dạng
├── training/                   # Công cụ chuẩn bị dữ liệu và train model
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

Chạy toàn bộ unit test:

```bash
python -m unittest discover -s tests -v
```

Chạy đánh giá theo tập ảnh và nhãn:

```bash
python batch_test.py <images_directory> <labels_directory>
```

Kết quả batch test được tổng hợp trong `batch_test_report.md`.

---

## ⚙️ Configuration

Các thiết lập chính nằm trong `app/core/config.py`:

| Thiết lập | Mặc định | Ý nghĩa |
|---|---:|---|
| `YOLO_MODEL_PATH` | `models/best.pt` | Model phát hiện biển số chính |
| `OCR_GPU` | `False` | Bật GPU cho EasyOCR |
| `OCR_FAST_MODE` | `True` | Ưu tiên tốc độ nhận diện |
| `OCR_DECODER` | `greedy` | Bộ giải mã EasyOCR |
| `MAX_DETECT_IMAGE_SIDE` | `960` | Giới hạn cạnh dài trước khi detect |
| `WARMUP_ON_STARTUP` | `True` | Warm-up model khi khởi động |

---

## 📸 Screenshots

<div align="center">

![Parking License Plate AI Dashboard](docs/screenshots/dashboard.png)

*Web Dashboard nhận diện biển số xe bằng YOLOv8 và EasyOCR.*

</div>

---

## 🚢 Deployment

Dự án hỗ trợ Docker và Hugging Face Spaces. Xem hướng dẫn chi tiết tại [HUGGINGFACE_DEPLOY.md](HUGGINGFACE_DEPLOY.md).

> Khi triển khai public, hãy cấu hình giới hạn kích thước file, authentication hoặc API gateway ở Backend để tránh lạm dụng tài nguyên AI.

---

## 📝 License & Contact

Hiện repository chưa khai báo giấy phép mã nguồn. Mọi quyền được bảo lưu cho nhóm phát triển cho đến khi file `LICENSE` được bổ sung.

- **Repository:** [ParkingBuildingManagementSystem-SWP391/ParkingBuildingLicensePlateAI](https://github.com/ParkingBuildingManagementSystem-SWP391/ParkingBuildingLicensePlateAI)
- **Organization:** [ParkingBuildingManagementSystem-SWP391](https://github.com/ParkingBuildingManagementSystem-SWP391)
- **Technical flow:** [LicensePlateRecognitionFlow.md](LicensePlateRecognitionFlow.md)

---

<div align="center">

Made with ❤️ for the **SWP391 Parking Building Management System**

⭐ Nếu dự án hữu ích, hãy đánh dấu sao cho repository!

</div>
