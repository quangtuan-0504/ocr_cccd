import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, List, Dict
from exceptions import (
    BadCaptureError,
    CardDetectionError,
    BadBoxError,
    ModelNotLoadedError
    )
from configs import settings

def validate_image_quality(image: Image.Image) -> None:
    """
    Kiểm tra tính hợp lệ của ảnh đầu vào.

    Args:
        image (Image.Image): Ảnh đầu vào định dạng PIL.

    Returns:
        bool: True nếu ảnh hợp lệ, False nếu không.

    Raises:
        ValueError: Nếu ảnh không hợp lệ (None hoặc không phải ảnh RGB).
    """
    
    if image is None:
        raise ValueError("Ảnh đầu vào không được để trống.")
    # Chuyển ảnh PIL sang numpy
    image_np = np.array(image)
    image_shape = image_np.shape
    # Kiểm tra kích thước ảnh
    height, width = image_shape[:2]
    if width < 200 or height < 200:
        raise BadCaptureError("Ảnh quá nhỏ")
    
    # Kiểm tra độ mờ bằng Laplacian variance
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_shape) == 3 else image_np
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 100:  # Threshold có thể điều chỉnh
        raise BadCaptureError("Ảnh quá mờ")
    # Kiểm tra độ sáng
    brightness = np.mean(gray) 
    if brightness < 50:  # Threshold có thể điều chỉnh
        raise BadCaptureError("Ảnh quá tối")
    # Kiểm tra độ tương phản
    contrast = gray.max() - gray.min()
    if contrast < 50:  # Threshold có thể điều chỉnh
        raise BadCaptureError("Ảnh có độ tương phản thấp")
    # Kiểm tra định dạng ảnh
    if len(image_shape) != 3 or image_shape[2] != 3 or image.mode != 'RGB':
        raise ValueError("Ảnh đầu vào phải là ảnh RGB.")
    


def validate_bounding_box(bbox:np.array , image_size: Tuple[int, int]) -> None:
    """
    Kiểm tra tính hợp lệ của bounding box
    """
    # tính diện tích box, ảnh
    img_width, img_height = image_size
    img_area = img_width * img_height
    
    x, y, box_width, box_height, rotate = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
    bbox_area = box_width * box_height
    area_ratio = bbox_area / img_area
    # check diện tích
    if area_ratio < settings.MIN_AREA_RATIO:
        raise BadBoxError(f"Diện tích thẻ quá nhỏ do bạn chụp quá xa: {area_ratio:.3f} < {settings.MIN_AREA_RATIO}")
    
    if area_ratio > settings.MAX_AREA_RATIO:
        raise BadBoxError(f"Diện tích thẻ quá lớn do bạn chụp quá gần: {area_ratio:.3f} > {settings.MAX_AREA_RATIO}")
    
    # Tính tỷ lệ chiều dài/rộng
    aspect_ratio = max(box_width, box_height) / min(box_width, box_height) # rộng / cao
    # Kiểm tra tỷ lệ khung hình
    if aspect_ratio < settings.ASPECT_RATIO_MIN or aspect_ratio > settings.ASPECT_RATIO_MAX:
        raise BadBoxError(f"Tỷ lệ chiều dài/rộng của thẻ không hợp lý do bạn chụp không chính diện: {aspect_ratio:.3f} not in [{settings.ASPECT_RATIO_MIN}, {settings.ASPECT_RATIO_MAX}]")

    
def crop_and_rotate_image(image: np.ndarray, box: np.ndarray):
        """
        Crop and rotate the image based on OBB box.

        Args:
            image: Input image as numpy array
            box: OBB box [x_center, y_center, width, height, angle]

        Returns:
            Cropped and rotated image as numpy array, or None if failed
        """
        try:
            x_center, y_center, width, height, angle = box
            angle = angle * 180 / np.pi  # Convert radians to degrees

            # Get rotation matrix
            M = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)

            # Rotate the image
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            # Crop the image
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            cropped = rotated[y1:y2, x1:x2]

            # Check if cropped image is valid
            if cropped.size == 0:
                return None

            return cropped
        except Exception as e:
            print(f"Error in cropping/rotating image: {str(e)}")
            return None

def determine_card_type_and_face(detections: List[BoundingBox]) -> Tuple[CardType, CardFace]:
    """
    Xác định loại thẻ và mặt thẻ dựa trên class_id
    Giả định:
    - class_id 0: CCCD mặt trước
    - class_id 1: CCCD mặt sau
    - class_id 2: CMT cũ
    """
    if not detections:
        return CardType.CCCD_2021, CardFace.FRONT
    
    # Lấy detection có confidence cao nhất
    best_detection = max(detections, key=lambda x: x.confidence)
    
    if best_detection.class_id == 0:
        return CardType.CCCD_2021, CardFace.FRONT
    elif best_detection.class_id == 1:
        return CardType.CCCD_2021, CardFace.BACK
    elif best_detection.class_id == 2:
        return CardType.CMT_OLD, CardFace.FRONT
    else:
        return CardType.CCCD_2021, CardFace.FRONT

def extract_field_from_text(text: str, field_type: str) -> str:
    """
    Trích xuất thông tin cụ thể từ text dựa trên loại trường
    """
    text = text.strip()
    
    if field_type == "id_number":
        # Trích xuất số CCCD/CMT
        import re
        numbers = re.findall(r'\d{9,12}', text)
        return numbers[0] if numbers else text
    
    elif field_type == "name":
        # Trích xuất tên
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return text
    
    elif field_type == "dob":
        # Trích xuất ngày sinh
        import re
        date_pattern = r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}'
        dates = re.findall(date_pattern, text)
        return dates[0] if dates else text
    
    return text

def parse_ocr_results(ocr_results: List, card_type: CardType, face: CardFace) -> List[dict]:
    """
    Parse kết quả OCR thành các trường có cấu trúc
    """
    fields = []
    
    for idx, result in enumerate(ocr_results):
        if len(result) >= 2:
            bbox_info = result[0]
            text_info = result[1]
            
            if isinstance(text_info, tuple) and len(text_info) >= 2:
                text = text_info[0]
                confidence = text_info[1]
                
                # Phân loại trường dựa vào nội dung text
                field_label = classify_field(text, card_type, face)
                field_value = extract_field_from_text(text, field_label.lower().replace(" ", "_"))
                
                fields.append({
                    "label": field_label,
                    "value": field_value,
                    "score": float(confidence)
                })
    
    return fields

def classify_field(text: str, card_type: CardType, face: CardFace) -> str:
    """
    Phân loại trường dữ liệu dựa trên nội dung text
    """
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in ["căn cước", "cccd", "số"]):
        return "Số CCCD"
    elif any(keyword in text_lower for keyword in ["họ", "tên", "name"]):
        return "Họ và tên"
    elif any(keyword in text_lower for keyword in ["sinh", "ngày", "date"]):
        return "Ngày sinh"
    elif any(keyword in text_lower for keyword in ["giới", "sex", "nam", "nữ"]):
        return "Giới tính"
    elif any(keyword in text_lower for keyword in ["quê", "quán", "origin"]):
        return "Quê quán"
    elif any(keyword in text_lower for keyword in ["thường", "trú", "address"]):
        return "Nơi thường trú"
    elif any(keyword in text_lower for keyword in ["cấp", "issue"]):
        return "Ngày cấp"
    elif any(keyword in text_lower for keyword in ["hết", "hạn", "expire"]):
        return "Có giá trị đến"
    else:
        return "Khác"