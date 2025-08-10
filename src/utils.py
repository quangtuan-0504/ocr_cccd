import numpy as np
import cv2
from PIL import Image, ExifTags, ImageOps
from typing import Optional, Tuple, List, Dict, Any, Union
from exceptions import (
    BadCaptureError,
    CardDetectionError,
    BadBoxError,
    ModelNotLoadedError,
    OCRBaseException
)
from configs import settings
from logger import logger
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from schemas import (
    format_res,
    cccd_map,
    cc_map
)
def validate_image_quality(image: Image.Image) -> None:
    """
    Validate the quality of the input image.

    Args:
        image: Input image in PIL format.

    Raises:
        ValueError: If the image is None or not an RGB image.
        BadCaptureError: If the image is too small, blurry, dark, or has low contrast.
    """
    if image is None:
        logger.error("Input image is None")
        raise ValueError("Input image cannot be empty.")
        
    # Convert PIL image to numpy array
    try:
        image_np = np.array(image)
    except Exception as e:
        logger.error(f"Failed to convert PIL image to numpy array: {e}")
        raise BadCaptureError(f"Invalid image format: {e}")
        
    image_shape = image_np.shape
    
    # Check image dimensions
    height, width = image_shape[:2]
    if width < 200 or height < 200:
        logger.warning(f"Image dimensions too small: {width}x{height}")
        raise BadCaptureError(f"Image is too small ({width}x{height}). Minimum size is 200x200 pixels.")
    
    # Check image format
    if len(image_shape) != 3 or image_shape[2] != 3 or image.mode != 'RGB':
        logger.warning(f"Invalid image format: shape={image_shape}, mode={image.mode}")
        raise ValueError(f"Input image must be in RGB format. Got shape={image_shape}, mode={image.mode}")
    
    try:
        # Check blur using Laplacian variance
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_shape) == 3 else image_np
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check brightness
        brightness = np.mean(gray) 
        
        # Check contrast
        contrast = gray.max() - gray.min()
        
        logger.info(f"Image quality metrics - blur: {laplacian_var:.2f}, brightness: {brightness:.2f}, contrast: {contrast:.2f}")
        
        if laplacian_var < 100:  # Adjustable threshold
            raise BadCaptureError(f"Image is too blurry (blur score: {laplacian_var:.2f})")
            
        if brightness < 50:  # Adjustable threshold
            raise BadCaptureError(f"Image is too dark (brightness: {brightness:.2f})")
            
        if contrast < 50:  # Adjustable threshold
            raise BadCaptureError(f"Image has low contrast (contrast: {contrast:.2f})")
    
    except OCRBaseException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error(f"Error during image quality validation: {e}")
        raise BadCaptureError(f"Failed to validate image quality: {e}")
    finally:
        del image_np  # Clean up numpy array to free memory
    
def correct_image_orientation(image: Image.Image) -> Image.Image:
    """
    Correct the orientation of an image based on EXIF data.
    
    Args:
        image: Input image in PIL format.
        
    Returns:
        Orientation-corrected image in PIL format.
    """
    if image is None:
        logger.error("Input image is None")
        raise ValueError("Input image cannot be empty.")
        
    try:
        # Get EXIF information
        if hasattr(image, '_getexif') and image._getexif() is not None:
            # Find orientation tag in EXIF data
            orientation_key = None
            for key in ExifTags.TAGS.keys():
                if ExifTags.TAGS[key] == 'Orientation':
                    orientation_key = key
                    break
                    
            exif = image._getexif()
            if exif is not None and orientation_key is not None:
                orientation = exif.get(orientation_key, 1)
                logger.info(f"EXIF orientation: {orientation}")
                
                # Rotate image based on orientation value
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                    logger.info("Rotated image 180 degrees")
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                    logger.info("Rotated image 270 degrees")
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
                    logger.info("Rotated image 90 degrees")
        
        # Auto-rotate based on image content if no EXIF data
        # This is an additional safety measure
        try:
            # Use ImageOps to auto-orient based on image content
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            logger.warning(f"Failed to auto-orient image: {e}")
                    
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image from {image.mode} to RGB")
            
        return image
        
    except Exception as e:
        logger.error(f"Error correcting image orientation: {e}")
        # Return original image if orientation correction fails
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

def validate_and_crop_card(xywh: np.ndarray,
                           conf: np.ndarray,
                           cls: np.ndarray,
                           image: np.ndarray,
                           expand: float = 1.1):
    """
    Validate detections: ensure 4 required corners and 1 card type. Optionally expand
    the corner geometry before perspective cropping.

    Args:
        xywh: (N,4) boxes [x_center, y_center, width, height]
        conf: (N,) confidences
        cls:  (N,) class ids
        image: source image array
        expand: scale factor to radially push corner centers away from card center (>1 enlarges)

    Returns:
        (cropped_image, w_ratio, h_ratio, card_type) or None if card type == 'other_card'

    Raises:
        CardDetectionError, BadBoxError
    """
    try:
        if any(v is None for v in (xywh, conf, cls, image)): # kiểm tra xem có giá trị nào là None không
            raise ValueError("Inputs cannot be None")
        if len(xywh) == 0:
            raise CardDetectionError("No detections found")

        corner_names = {'top_left', 'top_right', 'bottom_left', 'bottom_right'}
        card_type_classes = {'cc_front', 'cc_back', 'cccd_front', 'cccd_back', 'other_card'}

        # Keep highest confidence per relevant class
        best: Dict[str, Dict[str, Any]] = {}
        for i in range(len(xywh)):
            name = settings.CLASS_NAMES.get(int(cls[i]))
            if name is None:
                continue
            if name in corner_names or name in card_type_classes:
                if name not in best or conf[i] > best[name]['conf']: # chưa có thì thêm, có rồi thì so sánh xem có cập nhật lại k
                    best[name] = {
                        'box': xywh[i],
                        'conf': float(conf[i]),
                        'name': name
                    }

        # Check if we have exactly 5 required detections (4 corners + 1 card type)
        detected_corners = [name for name in corner_names if name in best]
        detected_card_types = [name for name in card_type_classes if name in best]
        
        # Must have exactly 4 corners
        if len(detected_corners) != 4:
            missing_corners = [c for c in corner_names if c not in best]# kiểm tra 4 góc thằng nào k có trong best thì thêm vào missing_corners
            if missing_corners:
                raise CardDetectionError(f"Missing corner detections: {missing_corners}")
            else:
                raise CardDetectionError("Unexpected corner detection issue")
        
        # Must have at least 1 card type
        if len(detected_card_types) == 0:
            raise CardDetectionError("Missing card type detection")

        # Select card type (highest conf among detected card-type classes)
        card_type_det = None
        for name in card_type_classes:
            if name in best:
                if card_type_det is None or best[name]['conf'] > card_type_det['conf']:
                    card_type_det = best[name]
        
        card_type_name = card_type_det['name']
        if card_type_name == 'other_card':
            raise CardDetectionError("Detected 'other_card' please sent another image.")

        def center_point(b):
            x_c, y_c, _, _ = b
            return np.array([float(x_c), float(y_c)], dtype=np.float32)

        # Get corner center coordinates
        corners_raw = {
            'top_left': center_point(best['top_left']['box']),
            'top_right': center_point(best['top_right']['box']),
            'bottom_right': center_point(best['bottom_right']['box']),
            'bottom_left': center_point(best['bottom_left']['box'])
        }

        # Calculate width and height based on corner centers
        # Width = distance from top_left to top_right
        width = float(np.linalg.norm(corners_raw['top_right'] - corners_raw['top_left'])) # tính tọa độ vector nối 2 điểm, xong tính độ dài của nó.
        # Height = distance from top_left to bottom_left  
        height = float(np.linalg.norm(corners_raw['bottom_left'] - corners_raw['top_left']))

        if width <= 0 or height <= 0:
            raise BadBoxError("Non-positive dimension after computation")

        # Validate aspect ratio
        aspect_ratio = max(width, height) / max(1e-6, min(width, height))
        if not (settings.ASPECT_RATIO_MIN <= aspect_ratio <= settings.ASPECT_RATIO_MAX):
            raise BadBoxError(f"Aspect ratio invalid: {aspect_ratio:.3f} not in "
                              f"[{settings.ASPECT_RATIO_MIN}, {settings.ASPECT_RATIO_MAX}]")

        # Validate area ratio
        img_h, img_w = image.shape[:2]
        card_area = width * height
        area_ratio = card_area / max(1, img_w * img_h)
        if area_ratio < settings.MIN_AREA_RATIO:
            raise BadBoxError(f"Card area too small: {area_ratio:.4f} < {settings.MIN_AREA_RATIO}")

        # Apply expansion if needed
        if expand != 1.0:
            # Get image dimensions
            img_h, img_w = image.shape[:2]
            
            # Calculate centroid of all corners
            pts_array = np.vstack([
                corners_raw['top_left'],
                corners_raw['top_right'], 
                corners_raw['bottom_right'],
                corners_raw['bottom_left']
            ])
            centroid = pts_array.mean(axis=0)
            
            # Expand corners away from centroid
            expanded_corners = {}
            for corner_name in corners_raw:
                expanded_corner = centroid + (corners_raw[corner_name] - centroid) * float(expand)
                
                # Clamp coordinates to image boundaries with small margin
                margin = 5  # pixels
                expanded_corner[0] = np.clip(expanded_corner[0], margin, img_w - margin)
                expanded_corner[1] = np.clip(expanded_corner[1], margin, img_h - margin)
                
                expanded_corners[corner_name] = expanded_corner
            
            # Update corners_raw with clamped expanded corners
            corners_raw = expanded_corners
            
            logger.debug(f"Applied expansion factor {expand} with image boundary constraints")
        # Prepare source and destination points for perspective transform
        src = np.array([
            corners_raw['top_left'],
            corners_raw['top_right'],
            corners_raw['bottom_right'],
            corners_raw['bottom_left']
        ], dtype=np.float32)

        # Destination rectangle (standard rectangle)
        dst_w = max(1, int(round(width)))
        dst_h = max(1, int(round(height)))
        dst = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype=np.float32)

        logger.debug(
            f"Geometry (expand={expand}) width={width:.1f}, height={height:.1f}, "
            f"aspect_ratio={aspect_ratio:.3f}, area_ratio={area_ratio:.4f}"
        )

        # Perspective transform
        M = cv2.getPerspectiveTransform(src, dst)
        cropped = cv2.warpPerspective(image, M, (dst_w, dst_h))

        w_ratio = width / img_w
        h_ratio = height / img_h
        logger.info(f"Cropped card type={card_type_name}, expand={expand}, w_ratio={w_ratio:.3f}, h_ratio={h_ratio:.3f}")
        return cropped, card_type_name, card_type_det['conf']

    except (BadBoxError, CardDetectionError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error in validate_and_crop_card: {e}")
        raise BadBoxError(f"Failed during card validation/cropping: {e}")

def format_date(date_str: str, code_gender_cen: str, date_type: str) -> str:
    year = date_str[:2]
    month = date_str[2:4]
    day = date_str[4:6]
    century = '20'
    if date_type == "dob":
        if code_gender_cen == '0' or code_gender_cen == '1':
            century = '19'
        elif code_gender_cen == '2'or code_gender_cen == '3':
            century = '20'
        elif code_gender_cen == '4'or code_gender_cen == '5':
            century = '21'
        elif code_gender_cen == '6'or code_gender_cen == '7':
            century = '22'
        elif code_gender_cen == '8'or code_gender_cen == '9':
            century = '23'
 
    return f"{day}/{month}/{century}{year}"
def parse_mrz(mrz: List) -> Dict[str, str]:
    res = {
        "id": mrz[0][15:27],
        "name": ' '.join(mrz[2][0:].replace('<', ' ').split()), # .split() tự động xóa ' ' dư thừa luôn.
        "doe": format_date(mrz[1][0:6],mrz[0][18], "doe"),
        "dob": format_date(mrz[1][0:6],mrz[0][18], "dob"),
        "nationality": mrz[0][2:5],
        "sex": mrz[1][7]
    }
    return res
def map_text2card(fieldname_text: Dict[str, Dict[str, Any]], require_type: str, require_side: str) -> Dict[str, Any]:
    """
    Map extracted text fields to card structure.
   
    Args:
        fieldname_text: Dictionary of field names and extracted text.
        require_type: Card type ('cccd' or 'cc').
        require_side: Card side ('front' or 'back').
       
    Returns:
        Dictionary with structured card information.
       
    Raises:
        ValueError: If the required card type or side is invalid.
    """
    # Validate inputs
    if not isinstance(fieldname_text, dict):
        logger.error(f"Invalid fieldname_text: {type(fieldname_text)}")
        raise ValueError("fieldname_text must be a dictionary")
       
    if require_type not in ['cccd', 'cc']:
        logger.error(f"Invalid card type: {require_type}")
        raise ValueError("require_type must be 'cccd' or 'cc'")
       
    if require_side not in ['front', 'back']:
        logger.error(f"Invalid card side: {require_side}")
        raise ValueError("require_side must be 'front' or 'back'")
   
    # print(fieldname_text)
   
    data = format_res.copy()
    if require_type == 'cc':
        field_mapping = cc_map.copy()
    elif require_type == 'cccd':
        field_mapping = cccd_map.copy()
    else:
        raise ValueError("Invalid require_type or require_side. Use 'cc' or 'cccd' for type and 'front' for side.")
   
    try:
        logger.info(f"Mapping text for {require_type} {require_side} side")
        logger.debug(f"Input fields: {list(fieldname_text.keys())}")
       
        # Duyệt qua fieldname_text để gán giá trị vào dictionary đầu ra
        average_score = 0
        cnt = 0
        for field, value in fieldname_text.items():
           
            if field in field_mapping:
                # Chọn tên trường tương ứng dựa trên require_type
                target_field = field_mapping[field]
                data[target_field] = value['text'] if isinstance(value['text'],str) else ''.join(value['text']) # mrz exception
                score = round(float(value['score']),2)
                data[target_field+'_prob'] = score
                average_score+=score
                cnt+=1
                if target_field == 'address':
                    address = value['text']
                    address_split = address.split(',')
                    address_split.reverse()
                    province, district, ward, *street = address_split
                    street.reverse()
                    street = ', '.join(street).strip()
                    data['custAddress']['province'] = province
                    data['custAddress']['district'] = district
                    data['custAddress']['ward'] = ward
                    data['custAddress']['street'] = street
                elif target_field == 'mrz':
                    data['mrz_details'] = parse_mrz(value['text'])
        if require_type == "cc" and require_side == 'front':
            data['type'] = 'cc_front'
            data['type_new'] = 'cccd_chip_front'
        elif require_type == "cc" and require_side == 'back':
            data['type'] = 'cc_back'
            data['type_new'] = 'chip_back'
        elif require_type == "cccd" and require_side == 'front':
            data['type'] = 'chip_front'
            data['type_new'] = 'cccd_chip_front'
        elif require_type == "cccd" and require_side == 'back':
            data['type'] = 'chip_front'
            data['type_new'] = 'chip_back'
        data['overall_score'] = round(average_score/cnt,2)
        logger.info(f"Successfully mapped {len(data)} fields")
        return data
       
    except Exception as e:
        logger.error(f"Error mapping text to card: {e}")
        # Return partial results if available, otherwise empty dict
        return data if 'street' in locals() else {}
