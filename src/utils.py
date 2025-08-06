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

def validate_bounding_box(bbox: np.ndarray, image_size: Tuple[int, int]) -> None:
    """
    Validate the bounding box of a detected card.
    
    Args:
        bbox: Bounding box array [x, y, width, height, rotation].
        image_size: Tuple of (width, height) of the original image.
        
    Raises:
        BadBoxError: If the bounding box is invalid (too small, too large, or wrong aspect ratio).
        ValueError: If the bbox or image_size parameters are invalid.
    """
    if bbox is None or len(bbox) != 5:
        logger.error(f"Invalid bounding box: {bbox}")
        raise ValueError("Bounding box must be a numpy array with 5 elements [x, y, width, height, rotation]")
        
    if len(image_size) != 2 or any(s <= 0 for s in image_size):
        logger.error(f"Invalid image size: {image_size}")
        raise ValueError("Image size must be a tuple of (width, height) with positive values")
    
    try:
        # Calculate areas
        img_width, img_height = image_size
        img_area = img_width * img_height
        
        x, y, box_width, box_height, rotate = bbox
        
        # Check for negative or zero dimensions
        if box_width <= 0 or box_height <= 0:
            raise BadBoxError(f"Invalid box dimensions: width={box_width}, height={box_height}")
        
        bbox_area = box_width * box_height
        area_ratio = bbox_area / img_area
        
        # Log the validation parameters
        logger.info(f"Box validation - area_ratio: {area_ratio:.3f}, box: {bbox}")
        
        # Check area ratio
        if area_ratio < settings.MIN_AREA_RATIO:
            raise BadBoxError(f"Card area is too small (taken from too far): {area_ratio:.3f} < {settings.MIN_AREA_RATIO}")
        
        # if area_ratio > settings.MAX_AREA_RATIO:
        #      raise BadBoxError(f"Card area is too large (taken from too close): {area_ratio:.3f} > {settings.MAX_AREA_RATIO}")
        
        
        # Calculate aspect ratio
        aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
        
        # Check aspect ratio
        if aspect_ratio < settings.ASPECT_RATIO_MIN or aspect_ratio > settings.ASPECT_RATIO_MAX:
            raise BadBoxError(f"Card aspect ratio is invalid (not taken from front): {aspect_ratio:.3f} not in [{settings.ASPECT_RATIO_MIN}, {settings.ASPECT_RATIO_MAX}]")
        # Compute width, height ratio
        w_ratio = box_width/img_width
        h_ratio = box_height/img_height
        return w_ratio, h_ratio

    except BadBoxError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error(f"Error validating bounding box: {e}")
        raise BadBoxError(f"Failed to validate bounding box: {e}")

def crop_and_rotate_image(image: np.ndarray, box: np.ndarray, expand_factor: float = 1.1):
    """
    Crop and rotate the image based on OBB box with customizable expansion.

    Args:
        image: Input image as numpy array
        box: OBB box [x_center, y_center, width, height, angle]
        expand_factor: Factor to expand the bounding box size (default 1.0 for original size)

    Returns:
        Cropped and rotated image as numpy array, or None if failed

    Nới rộng box theo tỷ lệ chiều rộng/dài
    """
    try:
        x_center, y_center, width, height, angle = box
        angle = angle * 180 / np.pi  # Convert radians to degrees
        # if width >= height:
        #     long_side = width
        #     short_side = height
        # else:
        #     long_side = height
        #     short_side = width

        # h_img, w_img, _ = image.shape
        # Apply expansion factor to width and height
        expanded_width = width * expand_factor
        expanded_height = height * expand_factor
        # if w_img/h_img>=0.5:
        #     expanded_width = expanded_width + expanded_width*0.2

        # Get rotation matrix
        M = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)

        # Rotate the image
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Calculate new coordinates with expanded dimensions
        x1 = int(x_center - expanded_width / 2)
        y1 = int(y_center - expanded_height / 2)
        x2 = int(x_center + expanded_width / 2)
        y2 = int(y_center + expanded_height / 2)

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
        raise(f"Error in cropping/rotating image: {str(e)}")
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
 