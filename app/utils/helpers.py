import re

from app.recognition.plate_rules import best_plate_candidate

# Bảng quy đổi tương đồng ký tự từ CHỮ sang SỐ (cho phân vùng bắt buộc là SỐ)
CHAR_TO_NUM = {
    'B': '8', 'S': '5', 'D': '0', 'O': '0', 'Q': '0', 
    'Z': '2', 'I': '1', 'T': '7', 'G': '6', 'A': '4', 
    'U': '0', 'L': '4', 'Y': '7', 'V': '7', 'F': '7', 
    'J': '1', 'H': '4', 'E': '3', 'C': '0', 'R': '8', 
    'P': '9'
}

# Bảng quy đổi tương đồng ký tự từ SỐ sang CHỮ (cho phân vùng bắt buộc là CHỮ)
NUM_TO_CHAR = {
    '0': 'D',
    '1': 'T',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'P'
}

# Bảng sửa đổi các chữ cái không hợp lệ trong sê-ri biển số Việt Nam (I, J, O, Q, W, R) sang chữ cái hợp lệ gần giống nhất
INVALID_CHAR_TO_VALID_CHAR = {
    'I': 'T',
    'J': 'T',
    'O': 'D',
    'Q': 'D',  # Ngoại lệ QT xử lý riêng
    'W': 'V',
    'R': 'B'
}

VALID_PROVINCE_CODES = {
    '11','12','14','15','16','17','18','19','20','21','22','23','24','25','26',
    '27','28','29','30','31','32','33','34','36','37','38','40','41','43','47',
    '48','49','50','51','52','53','54','55','56','57','58','59','60','61','62',
    '63','64','65','66','67','68','69','70','71','72','73','74','75','76','77',
    '78','79','80','81','82','83','84','85','86','88','89','90','92','93','94',
    '95','97','98','99'
}

# Hạ ngưỡng confidence tối thiểu xuống 0.08 để không bỏ sót dòng trên của xe máy
MIN_OCR_CONF = 0.20


def clean_motorbike_top_row(text: str, bottom_len: int = 5) -> str:
    """
    Làm sạch dòng trên của biển số xe máy (Ví dụ: '594D1' -> '59D1').
    Nhận diện và loại bỏ đinh ốc hoặc nét gạch ngang bị đọc thừa.
    """
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if len(text) < 3:
        return text

    # Xử lý phần số tỉnh đầu tiên (Ví dụ: 59)
    digits_part = ""
    for c in text[:2]:
        digits_part += CHAR_TO_NUM.get(c, c)
        
    # Tìm vị trí của chữ cái sê-ri chính, ưu tiên ký tự thực tế là chữ cái trước
    letter_idx = -1
    for i in range(2, len(text)):
        if text[i].isalpha():
            letter_idx = i
            break
            
    # Nếu không có ký tự nào là chữ cái thực sự, ta mới quy đổi các số tương đồng sang chữ
    if letter_idx == -1:
        for i in range(2, len(text)):
            c_char = NUM_TO_CHAR.get(text[i], text[i])
            if c_char.isalpha():
                letter_idx = i
                break
            
    if letter_idx == -1:
        if len(text) >= 4:
            return digits_part + text[2:4]
        return digits_part + text[2] + "1"
        
    series_char_raw = text[letter_idx]
    series_char = NUM_TO_CHAR.get(series_char_raw, series_char_raw)
    series_char = INVALID_CHAR_TO_VALID_CHAR.get(series_char, series_char)
    
    # Kiểm tra dịch chuyển sê-ri nếu chữ cái đầu là rác và kế tiếp là chữ cái khác
    if letter_idx + 1 < len(text):
        next_char_raw = text[letter_idx + 1]
        # Chỉ nhận chữ cái thực sự làm chữ cái dịch chuyển (loại bỏ lookalikes số)
        if next_char_raw.isalpha() and next_char_raw not in ['I', 'J', 'O', 'Q', 'W', 'R']:
            # Sê-ri 2 chữ cái xe máy chỉ tồn tại ở biển 5 số (bottom_len == 5) và bắt đầu bằng A hoặc M
            is_valid_two_letter_series = (bottom_len == 5) and (series_char in ['A', 'M'])
            if not is_valid_two_letter_series:
                # Nếu ký tự tiếp theo cũng là chữ cái, dịch chuyển sê-ri sang ký tự thứ hai
                letter_idx = letter_idx + 1
                series_char_raw = text[letter_idx]
                series_char = NUM_TO_CHAR.get(series_char_raw, series_char_raw)
                series_char = INVALID_CHAR_TO_VALID_CHAR.get(series_char, series_char)
            
    # Tìm chữ số sub-series hoặc chữ cái thứ hai nếu là sê-ri 2 chữ cái
    sub_series = ""
    for j in range(letter_idx + 1, len(text)):
        char_j = text[j]
        # Nếu ký tự tiếp theo là chữ cái, có thể đây là biển sê-ri 2 chữ cái (như AA, FB)
        # Chỉ nhận chữ cái hợp lệ thực sự làm chữ cái thứ hai (loại bỏ I, J, O, Q, W, R)
        if char_j.isalpha() and char_j not in ['I', 'J', 'O', 'Q', 'W', 'R']:
            # Nếu là ký tự cuối cùng của dòng trên, giữ nguyên làm chữ cái sê-ri thứ hai
            if j == len(text) - 1:
                sub_series = INVALID_CHAR_TO_VALID_CHAR.get(char_j, char_j)
                break
        
        # Ngược lại, xử lý như số thông thường
        if char_j.isdigit() and char_j != '0':
            sub_series = char_j
            break
        mapped_j = CHAR_TO_NUM.get(char_j, char_j)
        if mapped_j.isdigit() and mapped_j != '0':
            sub_series = mapped_j
            break
            
    if not sub_series:
        if letter_idx + 1 < len(text):
            sub_series = CHAR_TO_NUM.get(text[letter_idx + 1], text[letter_idx + 1])
        else:
            sub_series = "1"
            
    return digits_part + series_char + sub_series


def clean_motorbike_bottom_row(text: str, expected_len: int = None) -> str:
    """
    Làm sạch dòng dưới của biển số xe máy (Ví dụ: '462204' -> '46204').
    Quy đổi toàn bộ sang số và loại bỏ dấu chấm đọc nhầm thành chữ số.
    Args:
        expected_len: Số chữ số kỳ vọng của dòng dưới (4 hoặc 5), có được từ context dòng trên.
    """
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    digits_only = ""
    for c in text:
        digits_only += CHAR_TO_NUM.get(c, c)

    n = len(digits_only)

    if n <= 5:
        return digits_only

    if n == 6:
        # 1. Nếu có hai chữ số đầu tiên giống nhau và là 1 hoặc 7 (ví dụ: 112345 -> 12345)
        if digits_only[0] == digits_only[1] and digits_only[0] in ['1', '7']:
            return digits_only[1:]
        # 2. Nếu chữ số cuối cùng giống chữ số trước đó và là 1 hoặc 7
        if digits_only[5] == digits_only[4] and digits_only[5] in ['1', '7']:
            return digits_only[:5]
        # 3. Nếu chỉ chữ số đầu là nhiễu biên (1 hoặc 7) và chữ số cuối không phải
        if digits_only[0] in ['1', '7'] and digits_only[5] not in ['1', '7']:
            return digits_only[1:]
        # 4. Nếu chỉ chữ số cuối là nhiễu biên (1 hoặc 7) và chữ số đầu không phải
        if digits_only[5] in ['1', '7'] and digits_only[0] not in ['1', '7']:
            return digits_only[:5]
        # 5. Nếu cả hai đầu đều là 1 hoặc 7: kiểm tra xem có dấu chấm nhiễu ở giữa không
        if digits_only[0] in ['1', '7'] and digits_only[5] in ['1', '7']:
            if digits_only[3] in ['2', '8', '0']:
                return digits_only[:3] + digits_only[4:]
            else:
                return digits_only[1:]
        # 6. Mặc định loại bỏ chữ số thứ 4 (nhiễu dấu chấm)
        return digits_only[:3] + digits_only[4:]

    # 7 ký tự: giữ lại 5 ký tự có ý nghĩa nhất
    if n == 7:
        # Nếu 2 ký tự đầu giống nhau và là nhiễu -> bỏ 2 ký tự đầu, lấy 5 cuối
        if digits_only[0] == digits_only[1] and digits_only[0] in ['1', '7']:
            return digits_only[2:]
        # Nếu ký tự đầu là 1 hoặc 7 -> bỏ đầu, lấy 1-5
        if digits_only[0] in ['1', '7']:
            return digits_only[1:6]
        # Nếu ký tự cuối là 1 hoặc 7 -> bỏ cuối, lấy 1-5
        if digits_only[6] in ['1', '7']:
            return digits_only[1:6]
        return digits_only[1:6]

    # 8+ ký tự: lấy 5 ký tự đầu
    return digits_only[:5]


def clean_motorbike_digits(rem: str) -> str:
    if len(rem) <= 6:
        return rem
    if len(rem) == 8:
        rem = rem[1:5] + rem[6:]
    if len(rem) == 7:
        if rem[4] in ['2', '8']:
            return rem[:4] + rem[5:]
        elif rem[0] == '4':
            return rem[1:]
        elif rem[1] in ['1', '7', 'I']:
            return rem[0] + rem[2:]
        elif rem[0] in ['1', '7', 'I']:
            return rem[1:]
        else:
            return rem[:4] + rem[5:]
    return rem


def correct_split_characters(cleaned_text: str, is_motorbike: bool) -> str:
    """
    Sửa lỗi EasyOCR nhận diện một chữ cái rộng thành hai ký tự rời rạc ở cấp độ chuỗi.
    Chỉ thực hiện khi chiều dài chuỗi dài hơn bình thường để tránh sửa nhầm biển chuẩn.
    """
    # Nếu độ dài chuỗi <= 8, tuyệt đối không sửa lỗi tách ký tự vì có thể gây nhầm lẫn với cấu trúc chuẩn
    if len(cleaned_text) <= 8:
        return cleaned_text
        
    sub_2 = cleaned_text[2:4]
    
    # Các mẫu luôn luôn an toàn để sửa (chứa ký tự chắc chắn không hợp lệ ở vị trí tương ứng)
    always_safe_B = ['18', 'I8', '48', '4B']
    always_safe_U = ['40', '49', 'AO', '4O', '4U', 'AU']
    always_safe_H = ['11', 'II', 'HH']
    always_safe_C = ['71', '7I']
    
    # Các mẫu nhạy cảm (có thể là sê-ri hợp lệ trong cấu trúc chuẩn)
    sensitive_B = ['T8', 'L8', 'A8', 'A3', '43', 'AB']
    sensitive_U = ['A0', 'A9']
    sensitive_H = []
    sensitive_C = ['T1', 'TI']
    
    should_replace_B = sub_2 in always_safe_B
    should_replace_U = sub_2 in always_safe_U
    should_replace_H = sub_2 in always_safe_H
    should_replace_C = sub_2 in always_safe_C
    
    if is_motorbike:
        if not (should_replace_B or should_replace_U or should_replace_H or should_replace_C):
            if len(cleaned_text) == 10:
                should_replace_B = sub_2 in sensitive_B
                should_replace_U = sub_2 in sensitive_U
                should_replace_H = sub_2 in sensitive_H
                should_replace_C = sub_2 in sensitive_C
            elif len(cleaned_text) == 9:
                # Nếu sub_2 là A0 hoặc AO và ký tự sau đó là số (sê-ri xe máy không có số 0)
                if sub_2 in ['A0', 'AO']:
                    should_replace_U = True
    else:
        if not (should_replace_B or should_replace_U or should_replace_H or should_replace_C):
            if len(cleaned_text) == 10:
                should_replace_B = sub_2 in sensitive_B
                should_replace_U = sub_2 in sensitive_U
                should_replace_H = sub_2 in sensitive_H
                should_replace_C = sub_2 in sensitive_C
            elif len(cleaned_text) == 9:
                # Đối với ô tô: Nếu độ dài là 9 và ký tự thứ 4 (index 3) là số (không thể là biển 2 chữ cái)
                if cleaned_text[3].isdigit():
                    should_replace_B = sub_2 in sensitive_B
                    should_replace_U = sub_2 in sensitive_U
                    should_replace_H = sub_2 in sensitive_H
                    should_replace_C = sub_2 in sensitive_C
                    
    if should_replace_B:
        cleaned_text = cleaned_text[:2] + 'B' + cleaned_text[4:]
    elif should_replace_U:
        cleaned_text = cleaned_text[:2] + 'U' + cleaned_text[4:]
    elif should_replace_H:
        cleaned_text = cleaned_text[:2] + 'H' + cleaned_text[4:]
    elif should_replace_C:
        cleaned_text = cleaned_text[:2] + 'C' + cleaned_text[4:]
        
    return cleaned_text


def correct_license_plate_vietnam(text: str, is_motorbike: bool = False) -> str:
    """
    Hàm hậu xử lý sửa lỗi nhận diện biển số xe Việt Nam dựa trên vị trí ký tự.
    
    Tham số:
        text (str): Chuỗi ký tự thô đã được làm sạch và nối từ EasyOCR.
        is_motorbike (bool): Xe máy (True) hay ô tô (False).
        
    Trả về:
        str: Biển số đã sửa lỗi (Ví dụ: "30A12345", "59X312345").
    """
    if not text:
        return text

    # Loại bỏ khoảng trắng, dấu gạch ngang, chấm, phẩy và ký tự đặc biệt
    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()

    # Ưu tiên phân tích theo vị trí. Bộ phân tích này không chèn/xóa ký tự,
    # vì vậy không biến biển L0/M0 hợp lệ thành series LD/MD.
    structured_candidate = best_plate_candidate(cleaned_text)
    if structured_candidate is not None:
        return structured_candidate.text
    
    # Áp dụng chuẩn hóa ký tự bị tách đôi thông minh trước khi xử lý logic phân vùng
    cleaned_text = correct_split_characters(cleaned_text, is_motorbike)
    
    # Nếu là xe máy, tiến hành lọc bỏ ký tự rác (đinh ốc, dấu gạch ngang, dấu chấm đọc nhầm)
    if is_motorbike and len(cleaned_text) >= 9:
        prefix_series = cleaned_text[:3]
        rem = cleaned_text[3:]
        cleaned_rem = clean_motorbike_digits(rem)
        cleaned_text = prefix_series + cleaned_rem
        
    # Lọc bỏ ký tự rác cũ làm dự phòng
    if is_motorbike and len(cleaned_text) == 10 and cleaned_text[4] in ['1', '7', 'I']:
        cleaned_text = cleaned_text[:4] + cleaned_text[5:]
        
    # Nếu chuỗi quá ngắn không đủ cấu trúc biển số cơ bản, trả về luôn
    if len(cleaned_text) < 5:
        return cleaned_text

    # 1. Ký tự vị trí 1 và 2 (Mã tỉnh) -> Bắt buộc phải là SỐ
    char1 = CHAR_TO_NUM.get(cleaned_text[0], cleaned_text[0])
    char2 = CHAR_TO_NUM.get(cleaned_text[1], cleaned_text[1])
    prefix = (char1 if char1.isdigit() else cleaned_text[0]) + \
             (char2 if char2.isdigit() else cleaned_text[1])

    # Cảnh báo nếu mã tỉnh sửa xong vẫn không nằm trong danh sách mã tỉnh Việt Nam
    if prefix not in VALID_PROVINCE_CODES:
        print(f"[WARN] Mã tỉnh có thể không hợp lệ: '{prefix}' (gốc: '{cleaned_text[:2]}')")

    if is_motorbike:
        # ------------------ XE MÁY ------------------
        # Kiểm tra xem có phải biển số có sê-ri 2 chữ cái hay không (ví dụ: 59AA-12345, 51FB-4779)
        # Biển sê-ri 2 chữ cái xảy ra khi:
        # - Tổng chiều dài chuỗi là 8 (biển 4 số cũ: 51FB-4779) hoặc 9 (biển 5 số mới của xe 50cc: 59AA-12345)
        # - Và cả ký tự thứ 3 (index 2) và thứ 4 (index 3) đều là chữ cái
        c3_raw = cleaned_text[2]
        c4_raw = cleaned_text[3] if len(cleaned_text) >= 4 else ""
        
        c3 = NUM_TO_CHAR.get(c3_raw, c3_raw)
        c3_clean = INVALID_CHAR_TO_VALID_CHAR.get(c3, c3)
        
        c4 = NUM_TO_CHAR.get(c4_raw, c4_raw)
        c4_clean = INVALID_CHAR_TO_VALID_CHAR.get(c4, c4)
        
        is_double_letter_series = False
        if c3_clean.isalpha() and c4_clean.isalpha() and c3_raw.isalpha() and c4_raw.isalpha():
            if len(cleaned_text) == 8:
                is_double_letter_series = True
            elif len(cleaned_text) == 9 and c3_clean in ['A', 'M']:
                is_double_letter_series = True
                
        if is_double_letter_series:
            series_char = c3_clean
            sub_series = c4_clean
        else:
            series_char = c3_clean if c3_clean.isalpha() else c3
            # Sê-ri đơn thì ký tự thứ 4 bắt buộc phải là SỐ
            if len(cleaned_text) >= 4:
                char4 = CHAR_TO_NUM.get(c4_raw, c4_raw)
                sub_series = char4 if char4.isdigit() else c4_raw
            else:
                sub_series = ""
            
        # Toàn bộ phần còn lại (Đuôi số) -> Bắt buộc phải là SỐ
        tail = ""
        if len(cleaned_text) >= 5:
            # Nếu là sê-ri 2 chữ cái, phần đuôi số bắt đầu từ index 4. Ngược lại cũng bắt đầu từ index 4.
            for c in cleaned_text[4:]:
                mapped = CHAR_TO_NUM.get(c, c)
                tail += mapped if mapped.isdigit() else c
                
        return prefix + series_char + sub_series + tail

    else:
        # ------------------ Ô TÔ (Ví dụ: 30A - 12345 hoặc 30AA - 12345) ------------------
        is_two_letter_series = False
        
        if len(cleaned_text) >= 9:
            is_two_letter_series = True
        elif len(cleaned_text) == 8:
            c3_raw = cleaned_text[2]
            c4_raw = cleaned_text[3]
            
            # Chỉ coi là biển 2 chữ cái nếu ký tự thứ 4 (index 3) là chữ cái hoặc số '0' (thường nhầm với D)
            if c4_raw.isalpha() or c4_raw == '0':
                c3 = NUM_TO_CHAR.get(c3_raw, c3_raw)
                c4 = NUM_TO_CHAR.get(c4_raw, c4_raw)
                c3_clean = INVALID_CHAR_TO_VALID_CHAR.get(c3, c3)
                c4_clean = INVALID_CHAR_TO_VALID_CHAR.get(c4, c4)
                series_2 = (c3 if c3 == 'Q' and c4 == 'T' else c3_clean) + c4_clean
                if series_2 in ['LD', 'DA', 'NG', 'QT', 'MD']:
                    is_two_letter_series = True

        if is_two_letter_series:
            # Trường hợp: Biển ô tô 2 chữ cái (30AA...)
            c3_raw = cleaned_text[2]
            c4_raw = cleaned_text[3]
            
            c3 = NUM_TO_CHAR.get(c3_raw, c3_raw)
            c4 = NUM_TO_CHAR.get(c4_raw, c4_raw)
            
            s1 = INVALID_CHAR_TO_VALID_CHAR.get(c3, c3) if c3 != 'Q' else 'Q'
            s2 = INVALID_CHAR_TO_VALID_CHAR.get(c4, c4)
            
            if s1 == 'Q' and s2 != 'T':
                s1 = 'D'
                
            series_char = s1
            series_char_2 = s2
            
            # Đuôi số bắt đầu từ vị trí thứ 5 trở đi -> Bắt buộc phải là SỐ
            tail = ""
            if len(cleaned_text) >= 5:
                for c in cleaned_text[4:]:
                    mapped = CHAR_TO_NUM.get(c, c)
                    tail += mapped if mapped.isdigit() else c
                    
            return prefix + series_char + series_char_2 + tail
        else:
            # Trường hợp: Biển ô tô 1 chữ cái truyền thống (30A...)
            c3_raw = cleaned_text[2]
            c3 = NUM_TO_CHAR.get(c3_raw, c3_raw)
            series_char = INVALID_CHAR_TO_VALID_CHAR.get(c3, c3) if c3.isalpha() else c3
            
            # Đuôi số bắt đầu từ vị trí thứ 4 trở đi -> Bắt buộc phải là SỐ
            tail = ""
            if len(cleaned_text) >= 4:
                for c in cleaned_text[3:]:
                    mapped = CHAR_TO_NUM.get(c, c)
                    tail += mapped if mapped.isdigit() else c
                    
            return prefix + series_char + tail


def merge_close_boxes(line_boxes: list) -> list:
    """
    Gộp các bounding box nằm quá sát nhau trên cùng một dòng nằm ngang.
    Giải quyết triệt để lỗi EasyOCR nhận diện tách đôi một chữ cái rộng (như U thành A và 0, B thành 1 và 8).
    """
    if not line_boxes:
        return []

    # EasyOCR thường trả box cho cả cụm chữ. Gộp dựa trên khoảng cách đã làm
    # cặp số thật "1" + "8" thành "B", nên ở đây chỉ sắp xếp theo trục X.
    return sorted(line_boxes, key=lambda box: box["center_x"])
    
def preprocess_license_plate_text(ocr_results: list, is_motorbike: bool = False, crop_shape: tuple = None) -> str:
    """
    Sắp xếp các cụm từ đọc được từ EasyOCR theo dòng từ trên xuống dưới, từ trái sang phải,
    sau đó gửi sang hàm sửa lỗi logic biển số xe Việt Nam.
    """
    if not ocr_results:
        return ""

    # Lọc bỏ kết quả có độ tin cậy quá thấp và trích xuất thông tin tọa độ
    boxes_with_info = []
    for bbox, text, conf in ocr_results:
        if conf < MIN_OCR_CONF:
            print(f"[OCR FILTER] Bỏ qua '{text}' vì conf={conf:.2f} < {MIN_OCR_CONF}")
            continue

        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        center_y = (min_y + max_y) / 2
        center_x = (min_x + max_x) / 2
        height = max_y - min_y
        width = max_x - min_x

        boxes_with_info.append({
            "text": text,
            "center_y": center_y,
            "center_x": center_x,
            "min_x": min_x,
            "max_x": max_x,
            "width": width,
            "height": height,
            "conf": conf,
        })

    if not boxes_with_info:
        return ""

    # PHÂN DÒNG TỐI ƯU (SỬ DỤNG MIDDLE-LINE SPLIT NẾU CÓ CROP_SHAPE)
    lines = []
    if crop_shape is not None and len(crop_shape) >= 2:
        h_ocr, w_ocr = crop_shape[0], crop_shape[1]
        ratio = w_ocr / h_ocr
        
        # Nếu tỉ lệ rộng/cao < 2.0 -> Biển vuông (xe máy hoặc ô tô vuông) có 2 dòng
        if ratio < 2.0:
            top_line = []
            bottom_line = []
            ocr_middle_y = h_ocr / 2
            
            for box in boxes_with_info:
                if box["center_y"] < ocr_middle_y:
                    top_line.append(box)
                else:
                    bottom_line.append(box)
            
            # Sắp xếp các từ cùng dòng từ trái qua phải
            top_line.sort(key=lambda b: b["center_x"])
            bottom_line.sort(key=lambda b: b["center_x"])
            
            if top_line:
                lines.append(top_line)
            if bottom_line:
                lines.append(bottom_line)
        else:
            # Biển dài (1 dòng) -> Sắp xếp toàn bộ từ trái qua phải
            boxes_with_info.sort(key=lambda b: b["center_x"])
            lines.append(boxes_with_info)
    else:
        # Fallback thuật toán cũ nếu không có thông tin crop_shape (Dynamic Y-clustering)
        # Bước A1: Sắp xếp các box theo trục Y từ trên xuống dưới
        boxes_with_info.sort(key=lambda b: b["center_y"])

        # Bước A2: Gom nhóm thành các dòng dựa vào chiều cao của chữ
        current_line = [boxes_with_info[0]]
        for box in boxes_with_info[1:]:
            avg_height = sum(b["height"] for b in current_line) / len(current_line)
            avg_y = sum(b["center_y"] for b in current_line) / len(current_line)

            # Nếu khoảng cách lệch Y nhỏ hơn 0.8 lần chiều cao chữ, coi như cùng 1 dòng
            if abs(box["center_y"] - avg_y) < (avg_height * 0.8):
                current_line.append(box)
            else:
                # Sắp xếp các từ cùng dòng từ trái qua phải (trục X)
                current_line.sort(key=lambda b: b["center_x"])
                lines.append(current_line)
                current_line = [box]

        # Lưu dòng cuối cùng
        current_line.sort(key=lambda b: b["center_x"])
        lines.append(current_line)

    # Gộp các box quá sát nhau trên từng dòng (chống tách chữ)
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(merge_close_boxes(line))
    lines = cleaned_lines

    # Nếu là xe máy và phát hiện 2 dòng riêng biệt rõ ràng
    if is_motorbike and len(lines) >= 2:
        raw_top = "".join(box["text"] for box in lines[0])
        raw_bottom = "".join(box["text"] for box in lines[1])

        clean_bottom = clean_motorbike_bottom_row(raw_bottom)
        clean_top = clean_motorbike_top_row(raw_top, len(clean_bottom))
        
        combined = clean_top + clean_bottom
        candidate = best_plate_candidate(combined)
        return candidate.text if candidate is not None else combined

    # Nối tất cả các cụm từ theo dòng đã sắp xếp
    raw_text = "".join(box["text"] for line in lines for box in line)

    # Chạy hàm hậu xử lý sửa lỗi theo quy định biển số xe Việt Nam
    return correct_license_plate_vietnam(raw_text, is_motorbike)
