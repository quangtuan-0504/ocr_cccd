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
        
        if area_ratio > settings.MAX_AREA_RATIO:
            raise BadBoxError(f"Card area is too large (taken from too close): {area_ratio:.3f} > {settings.MAX_AREA_RATIO}")
        
        # Calculate aspect ratio
        aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
        
        # Check aspect ratio
        if aspect_ratio < settings.ASPECT_RATIO_MIN or aspect_ratio > settings.ASPECT_RATIO_MAX:
            raise BadBoxError(f"Card aspect ratio is invalid (not taken from front): {aspect_ratio:.3f} not in [{settings.ASPECT_RATIO_MIN}, {settings.ASPECT_RATIO_MAX}]")
    
    except BadBoxError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error(f"Error validating bounding box: {e}")
        raise BadBoxError(f"Failed to validate bounding box: {e}")

def crop_and_rotate_image(image: np.ndarray, box: np.ndarray, expand_factor: float = 1.05) -> np.ndarray:
    """
    Crop and rotate the image based on the bounding box.
    
    Args:
        image: Input image as numpy array.
        box: Bounding box array [x, y, width, height, rotation].
        expand_factor: Factor to expand the bounding box (default: 1.05).
        
    Returns:
        Cropped and rotated image as numpy array.
        
    Raises:
        CardDetectionError: If cropping or rotation fails.
        ValueError: If input parameters are invalid.
    """
    if image is None or not isinstance(image, np.ndarray):
        logger.error("Invalid image input")
        raise ValueError("Input image must be a numpy array")
        
    if box is None or len(box) != 5:
        logger.error(f"Invalid bounding box: {box}")
        raise ValueError("Bounding box must be a numpy array with 5 elements [x, y, width, height, rotation]")
    
    try:
        # Extract box parameters
        cx, cy, w, h, angle = box
        
        # Check for invalid dimensions
        if w <= 0 or h <= 0:
            raise CardDetectionError(f"Invalid box dimensions: width={w}, height={h}")
            
        # Check if center is within image bounds
        img_height, img_width = image.shape[:2]
        if not (0 <= cx < img_width and 0 <= cy < img_height):
            logger.warning(f"Box center ({cx}, {cy}) outside image bounds ({img_width}, {img_height})")
        
        # Expand box dimensions
        w *= expand_factor
        h *= expand_factor
        
        # Calculate the rotation matrix
        angle_rad = np.deg2rad(angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        # Calculate the four corners of the rotated rectangle
        x1 = cx - w/2 * cos_theta + h/2 * sin_theta
        y1 = cy - w/2 * sin_theta - h/2 * cos_theta
        
        x2 = cx + w/2 * cos_theta + h/2 * sin_theta
        y2 = cy + w/2 * sin_theta - h/2 * cos_theta
        
        x3 = cx + w/2 * cos_theta - h/2 * sin_theta
        y3 = cy + w/2 * sin_theta + h/2 * cos_theta
        
        x4 = cx - w/2 * cos_theta - h/2 * sin_theta
        y4 = cy - w/2 * sin_theta + h/2 * cos_theta
        
        # Define source points (rotated rectangle corners)
        src_pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
        
        # Calculate width and height of the destination rectangle
        dst_width = int(w)
        dst_height = int(h)
        
        # Define destination points (straight rectangle corners)
        dst_pts = np.array([[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]], dtype=np.float32)
        
        # Get perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply perspective transformation
        cropped_img = cv2.warpPerspective(image, M, (dst_width, dst_height))
        
        # Check if the cropped image is valid
        if cropped_img is None or cropped_img.size == 0:
            raise CardDetectionError("Failed to crop image: empty result")
            
        logger.info(f"Successfully cropped image to size: {cropped_img.shape}")
        return cropped_img
        
    except CardDetectionError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error(f"Error in crop_and_rotate_image: {e}")
        raise CardDetectionError(f"Failed to crop and rotate image: {e}")

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
    
    try:
        # Define dictionaries for cc and cccd
        result = {}
        
        logger.info(f"Mapping text for {require_type} {require_side} side")
        logger.debug(f"Input fields: {list(fieldname_text.keys())}")
        
        # Process the fields based on card type and side
        if require_type == 'cc':
            if require_side == 'front':
                for field in fieldname_text:
                    if field == 'id':
                        result['id'] = fieldname_text[field]['text']
                    elif field == 'name':
                        result['name'] = fieldname_text[field]['text']
                    elif field == 'date_of_birth':
                        result['date_of_birth'] = fieldname_text[field]['text']
                    elif field == 'sex':
                        result['sex'] = fieldname_text[field]['text']
                    elif field == 'nation':
                        result['nation'] = fieldname_text[field]['text']
                    elif field == 'place_of_residence':
                        result['place_of_residence'] = fieldname_text[field]['text']
            else:  # back side
                for field in fieldname_text:
                    if field == 'place_of_birth':
                        result['place_of_birth'] = fieldname_text[field]['text']
                    elif field == 'date_of_issue':
                        result['date_of_issue'] = fieldname_text[field]['text']
                    elif field == 'date_of_expiry':
                        result['date_of_expiry'] = fieldname_text[field]['text']
        
        elif require_type == 'cccd':
            if require_side == 'front':
                for field in fieldname_text:
                    if field == 'id':
                        result['id'] = fieldname_text[field]['text']
                    elif field == 'name':
                        result['name'] = fieldname_text[field]['text']
                    elif field == 'date_of_birth':
                        result['date_of_birth'] = fieldname_text[field]['text']
                    elif field == 'gender':
                        result['gender'] = fieldname_text[field]['text']
                    elif field == 'nation':
                        result['nation'] = fieldname_text[field]['text']
                    elif field == 'hometown':
                        result['hometown'] = fieldname_text[field]['text']
                    elif field == 'permanent_residence':
                        result['permanent_residence'] = fieldname_text[field]['text']
            else:  # back side
                for field in fieldname_text:
                    if field == 'date_of_expiry':
                        result['date_of_expiry'] = fieldname_text[field]['text']
        
        logger.info(f"Successfully mapped {len(result)} fields")
        return result
        
    except Exception as e:
        logger.error(f"Error mapping text to card: {e}")
        # Return partial results if available, otherwise empty dict
        return result if 'result' in locals() else {}