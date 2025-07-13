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

def map_text2card(fieldname_text, require_type, require_side):
    # Định nghĩa các dictionary cho cc và cccd
    cc_front_fields = {
        "Số định danh cá nhân / Personal identification number": "",
        "Họ, chữ đệm và tên khai sinh / Full name": "",
        "Ngày, tháng, năm sinh / Date of birth": "",
        "Giới tính / Sex": "",
        "Quốc tịch / Nationality": ""
    }
    cccd_front_fields = {
        "Số / No.": "",
        "Họ và tên / Full name": "",
        "Ngày sinh / Date of birth": "",
        "Giới tính / Sex": "",
        "Quốc tịch / Nationality": "",
        "Quê quán / Place of origin": "",
        "Nơi thường trú / Place of residence": "",
        "Có giá trị đến / Date of expiry": ""
    }
    
    # Chọn dictionary đầu ra dựa trên require_type và require_side
    if require_type == 'cc' and require_side == 'front':
        output_dict = cc_front_fields.copy()
    elif require_type == 'cccd' and require_side == 'front':
        output_dict = cccd_front_fields.copy()
    else:
        raise ValueError("Invalid require_type or require_side. Use 'cc' or 'cccd' for type and 'front' for side.")
    
    # Ánh xạ từ FIELD_CLASS sang tên trường trong dictionary đầu ra
    field_mapping = {
        'id number': ["Số định danh cá nhân / Personal identification number", "Số / No."],
        'full name': ["Họ, chữ đệm và tên khai sinh / Full name", "Họ và tên / Full name"],
        'date of birth': ["Ngày, tháng, năm sinh / Date of birth", "Ngày sinh / Date of birth"],
        'sex': ["Giới tính / Sex", "Giới tính / Sex"],
        'nationality': ["Quốc tịch / Nationality", "Quốc tịch / Nationality"],
        'place of origin': ["", "Quê quán / Place of origin"],
        'place of residence': ["", "Nơi thường trú / Place of residence"],
        'date of expiry': ["", "Có giá trị đến / Date of expiry"]
    }
    
    # Duyệt qua fieldname_text để gán giá trị vào dictionary đầu ra
    for field, value in fieldname_text:
        if field in field_mapping:
            # Chọn tên trường tương ứng dựa trên require_type
            target_field = field_mapping[field][0] if require_type == 'cc' else field_mapping[field][1]
            if target_field and target_field in output_dict:
                # Gán value
                output_dict[target_field] = value
    
    return output_dict