import os
import sys
import cv2
import re

# Thêm thư mục gốc vào path để import được app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.services.detector import detector_service
except ImportError:
    print("Không thể import detector_service. Hãy đảm bảo bạn chạy script này từ thư mục gốc của dự án.")
    sys.exit(1)

def clean_text(text):
    if not text:
        return ""
    # Chỉ giữ chữ và số, viết hoa
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()

def run_batch_test(images_dir, labels_dir, is_motorbike=True):
    if not os.path.exists(images_dir):
        print(f"Lỗi: Thư mục chứa ảnh không tồn tại: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"Lỗi: Thư mục chứa label không tồn tại: {labels_dir}")
        return

    # Danh sách các định dạng ảnh được hỗ trợ
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    
    total = 0
    correct = 0
    mismatches = []
    
    print(f"Tìm thấy {len(image_files)} file ảnh trong thư mục '{images_dir}'.")
    print("Đang tiến hành chạy nhận diện hàng loạt...\n")
    
    for i, file_name in enumerate(image_files, 1):
        img_path = os.path.join(images_dir, file_name)
        base_name, _ = os.path.splitext(file_name)
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        # Đọc ground truth từ file label
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r", encoding="utf-8") as f:
            gt_text = clean_text(f.read().strip())
            
        if not gt_text:
            continue
            
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{i}/{len(image_files)}] Lỗi đọc file ảnh: {file_name}")
            continue
            
        # Chạy nhận diện
        try:
            recognized_raw, _, _ = detector_service.detect_and_recognize(img, is_motorbike=is_motorbike)
            recognized = clean_text(recognized_raw)
        except Exception as e:
            recognized = ""
            print(f"[{i}/{len(image_files)}] Lỗi khi nhận diện file {file_name}: {str(e)}")
            
        total += 1
        is_match = (recognized == gt_text)
        
        if is_match:
            correct += 1
            print(f"[{i}/{len(image_files)}] Ảnh {file_name}: Nhãn gốc: {gt_text} | AI đọc: {recognized} -> OK")
        else:
            mismatches.append({
                "file_name": file_name,
                "expected": gt_text,
                "got": recognized
            })
            print(f"[{i}/{len(image_files)}] Ảnh {file_name}: Nhãn gốc: '{gt_text}' | AI đọc: '{recognized}' -> SAI")
            
    # Hiển thị báo cáo kết quả
    accuracy = (correct / total * 100) if total > 0 else 0.0
    print("\n" + "="*60)
    print(" KẾT QUẢ KIỂM THỬ HÀNG LOẠT (BATCH TEST REPORT)")
    print("="*60)
    print(f"Tổng số ảnh có nhãn kiểm thử: {total}")
    print(f"Nhận diện đúng: {correct}")
    print(f"Nhận diện sai: {len(mismatches)}")
    print(f"Tỷ lệ chính xác (Accuracy): {accuracy:.2f}%")
    print("="*60)
    
    # Ghi báo cáo ra file markdown
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Kết Quả Nhận Diện Biển Số Xe Hàng Loạt\n\n")
        f.write(f"- **Thư mục ảnh**: `{images_dir}`\n")
        f.write(f"- **Thư mục nhãn**: `{labels_dir}`\n")
        f.write(f"- **Tổng số ảnh kiểm thử**: {total}\n")
        f.write(f"- **Số lượng đúng**: {correct} / {total}\n")
        f.write(f"- **Số lượng sai**: {len(mismatches)} / {total}\n")
        f.write(f"- **Tỷ lệ chính xác**: `{accuracy:.2f}%`\n\n")
        
        if mismatches:
            f.write("## Chi Tiết Các Trường Hợp Nhận Diện Sai\n\n")
            f.write("| STT | Tên File | Biển Gốc (Kỳ Vọng) | Nhận Diện Thực Tế | Trạng Thái |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for idx, item in enumerate(mismatches, 1):
                got_display = f"`{item['got']}`" if item['got'] else "*Không phát hiện biển*"
                f.write(f"| {idx} | {item['file_name']} | `{item['expected']}` | {got_display} | ❌ Sai |\n")
        else:
            f.write("🎉 Tuyệt vời! Tất cả các ảnh đều được nhận diện chính xác 100%!\n")
            
    print(f"Đã ghi báo cáo chi tiết vào file: {report_path}")

if __name__ == "__main__":
    # Đường dẫn mặc định trỏ tới thư mục của User
    default_images = r"C:\Users\Admin\Downloads\biensoxemayhon100bien\anh"
    default_labels = r"C:\Users\Admin\Downloads\biensoxemayhon100bien\label"
    
    # Cho phép ghi đè đường dẫn qua tham số terminal
    img_dir = sys.argv[1] if len(sys.argv) > 1 else default_images
    lbl_dir = sys.argv[2] if len(sys.argv) > 2 else default_labels
    
    run_batch_test(img_dir, lbl_dir, is_motorbike=True)
