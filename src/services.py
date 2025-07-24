import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from PIL import Image
import cv2
from paddleocr import DocImgOrientationClassification, TextImageUnwarping
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
import time
import concurrent.futures
from functools import lru_cache

from configs import settings
from utils import (
    validate_image_quality,
    validate_bounding_box,
    crop_and_rotate_image,
    map_text2card,
    correct_image_orientation
)
from logger import logger
from exceptions import CardDetectionError, BadCaptureError

class IdCardOCRService:
    """Service for processing ID card images using YOLOv11 OBB, PaddleOCR and VietOCR."""
    
    # Card class definitions
    CLASS_CARD_NAMES = settings.CLASS_NAMES
    CLASS_FIELD_NAME = settings.FIELD_CLASS
    
    def __init__(
        self,
        cardobb_model: YOLO,
        imgorient_model: DocImgOrientationClassification,
        imgunwarp_model: TextImageUnwarping,
        textdetcls_model: Dict[str, YOLO],
        rec_model: Predictor
    ):
        """Initialize the ID card OCR service.

        Args:
            cardobb_model: YOLOv11 OBB model for card detection and cropping.
            imgorient_model: Model for image orientation classification.
            imgunwarp_model: Model for text image unwarping.
            textdetcls_model: Dictionary of YOLO models for text detection and classification.
            rec_model: VietOCR model for text recognition.
        """
        self.cardobb_model = cardobb_model
        self.imgorient_model = imgorient_model
        self.imgunwarp_model = imgunwarp_model
        self.textdetcls_model = textdetcls_model
        self.rec_model = rec_model
        
        # Initialize thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Cache for frequently accessed operations
        self.cache_size = 128  # Adjust based on memory constraints
        
        # Apply caching to expensive operations
        self._cached_rotate_card = lru_cache(maxsize=self.cache_size)(self._rotate_card)
        self._cached_unwarp_card = lru_cache(maxsize=self.cache_size)(self._unwarp_card)

    def _predict_card(self, image: Image.Image) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Detect and crop the card in the image using YOLOv11 OBB.

        Args:
            image: Input image in PIL format.

        Returns:
            Tuple containing the cropped image (if any) and prediction information.
        """
        try:
            # Convert PIL image to numpy array once
            image_np = np.array(image)
            
            # Use batched inference with optimized settings
            inference_config = settings.yolo_cardobb_configs["inference"].copy()
            inference_config["verbose"] = False  # Reduce logging overhead
            
            results = self.cardobb_model.predict(image, **inference_config)

            if not results or not results[0].obb:
                return None, {"error": "No card detected in image"}

            obb = results[0].obb
            boxes = obb.xywhr.cpu().numpy()
            scores = obb.conf.cpu().numpy()
            classes = obb.cls.cpu().numpy().astype(int)

            # Find the detection with highest confidence
            max_conf_idx = scores.argmax()
            best_box = boxes[max_conf_idx]
            best_class = classes[max_conf_idx]
            best_score = scores[max_conf_idx]

            img_height, img_width = image_np.shape[:2]
            validate_bounding_box(bbox=best_box, image_size=(img_width, img_height))
            cropped_image = crop_and_rotate_image(image_np, best_box)

            logger.info(f"Detected card class: {self.CLASS_CARD_NAMES[best_class]} with confidence {best_score:.3f}")
            return cropped_image, {
                "class": self.CLASS_CARD_NAMES[best_class],
                "confidence": float(best_score)
            }

        except Exception as e:
            logger.error(f"Error predicting card: {str(e)}")
            return None, {"error": f"Card prediction failed: {str(e)}"}

    def _rotate_card(self, cropped_image: np.ndarray) -> np.ndarray:
        """Rotate the cropped image based on predicted orientation.

        Args:
            cropped_image: Input image as a numpy array.

        Returns:
            Rotated image as a numpy array.
        """
        # Calculate hash of input for caching
        image_hash = hash(cropped_image.tobytes())
        
        output = self.imgorient_model.predict(
            input=cropped_image, 
            **settings.paddle_imgorient_configs['inference']
        )
        orientation = output[0]['label_names'][0]

        logger.info(f"Predicted orientation: {orientation}")
        if orientation == "0":
            return cropped_image
        elif orientation == "180":
            return cv2.rotate(cropped_image, cv2.ROTATE_180)
        elif orientation == "90":
            return cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == "270":
            return cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        else:
            logger.warning(f"Unknown orientation: {orientation}")
            return cropped_image
    
    def _unwarp_card(self, rotated_image: np.ndarray) -> np.ndarray:
        """Unwarp the rotated image to correct text distortion.

        Args:
            rotated_image: Rotated image as a numpy array.

        Returns:
            Unwarped image as a numpy array.
        """
        output = self.imgunwarp_model.predict(
            input=rotated_image, 
            **settings.paddle_imgunwarp_configs['inference']
        )
        unwarped_img = output[0]['doctr_img']
        logger.info(f"Unwarped image shape: {unwarped_img.shape}")
        return unwarped_img

    def _detect_cls_text(self, rotated_unwarped_img: np.ndarray, required_type: str) -> List[Dict[str, Any]]:
        """Detect and classify text regions in the unwarped image.

        Args:
            rotated_unwarped_img: Unwarped image as a numpy array.
            required_type: Type of card ("cccd" or "cc").

        Returns:
            List of dictionaries containing field, text region, score, and y_min.
        """
        img_pil = Image.fromarray(rotated_unwarped_img)
        
        # Use optimized inference settings
        inference_config = settings.yolo_textdetcls_configs[required_type]['inference'].copy()
        inference_config["verbose"] = False
        inference_config["save"] = False  # Don't save images to disk
        
        results = self.textdetcls_model[required_type].predict(
            source=img_pil, **inference_config
        )

        fieldname_textregion_score = []
        for result in results:
            boxes = result.boxes
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            detected_fields = [self.CLASS_FIELD_NAME[required_type][int(cls)] for cls in classes]
            logger.info(f"Detected fields: {detected_fields}")

            # Process all detected regions at once
            for j, box in enumerate(boxes_xyxy):
                x_min, y_min, x_max, y_max = map(int, box)
                # Ensure coordinates are within bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(rotated_unwarped_img.shape[1], x_max)
                y_max = min(rotated_unwarped_img.shape[0], y_max)
                
                # Skip invalid regions
                if x_min >= x_max or y_min >= y_max:
                    logger.warning(f"Skipping invalid text region: {x_min},{y_min},{x_max},{y_max}")
                    continue
                    
                cropped_img = rotated_unwarped_img[y_min:y_max, x_min:x_max]
                fieldname_textregion_score.append({
                    'field': classes[j],
                    'text_region': cropped_img,
                    'score': scores[j],
                    'y_min': y_min
                })

        return fieldname_textregion_score

    def _extract_text_region(self, region_data: Dict[str, Any], required_type: str) -> Dict[str, Any]:
        """Extract text from a single region using VietOCR.
        
        Args:
            region_data: Dictionary containing field data and image region.
            required_type: Type of card ("cccd" or "cc").
            
        Returns:
            Dictionary with field name, extracted text and metadata.
        """
        cls = region_data['field']
        str_cls = self.CLASS_FIELD_NAME[required_type][cls]
        y_min = region_data['y_min']
        score = region_data['score']
        
        try:
            # Convert image for OCR
            cropped_img_pil = Image.fromarray(cv2.cvtColor(region_data['text_region'], cv2.COLOR_BGR2RGB))
            vietocr_text = self.rec_model.predict(cropped_img_pil)
            
            logger.info(f"Extracted text for field {str_cls}: {vietocr_text}")
            return {
                'field_name': str_cls,
                'text': vietocr_text,
                'y_min': y_min,
                'score': score
            }
        except Exception as e:
            logger.error(f"Error extracting text for field {str_cls}: {e}")
            return {
                'field_name': str_cls,
                'text': '',
                'y_min': y_min,
                'score': 0.0,
                'error': str(e)
            }

    def _extract_text(self, fieldname_textregion_score: List[Dict[str, Any]], required_type: str) -> Dict[str, Dict[str, Any]]:
        """Extract text from detected regions using VietOCR with parallel processing.

        Args:
            fieldname_textregion_score: List of detected text regions with metadata.
            required_type: Type of card ("cccd" or "cc").

        Returns:
            Dictionary mapping field names to extracted text and metadata.
        """
        try:
            if not fieldname_textregion_score:
                logger.warning("No text regions to extract")
                return {}
                
            # Use parallel processing for text extraction
            futures = []
            for region_data in fieldname_textregion_score:
                futures.append(
                    self.executor.submit(
                        self._extract_text_region, 
                        region_data, 
                        required_type
                    )
                )
            
            # Collect results
            field_text = {}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if 'error' in result:
                    logger.warning(f"Text extraction issue: {result.get('error', 'Unknown error')}")
                    continue
                    
                str_cls = result['field_name']
                text = result['text']
                y_min = result['y_min']
                score = result['score']
                
                if str_cls in field_text:
                    if y_min > field_text[str_cls]['y_min']:
                        field_text[str_cls]['text'] += ' ' + text
                    else:
                        field_text[str_cls]['text'] = text + ' ' + field_text[str_cls]['text']
                    field_text[str_cls]['score'] = (field_text[str_cls]['score'] + score) / 2
                else:
                    field_text[str_cls] = {'text': text, 'y_min': y_min, 'score': score}

            return field_text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {"error": f"Text extraction failed: {str(e)}"}

    def process_id_card(self, image: Image.Image, required_type: str = "cccd", required_side: str = "front") -> Dict[str, Any]:
        """Process an ID card image and extract information.

        Args:
            image: Input image in PIL format.
            required_type: Card type ("cccd" or "cc").
            required_side: Card side ("front" or "back").

        Returns:
            Dictionary containing card information and extracted text or error message.
        """
        try:
            # Validate and preprocess image
            start_time = time.time()
            validate_image_quality(image)
            image = correct_image_orientation(image)
            logger.info(f'Time for verify image: {time.time() - start_time:.3f}s')

            # Detect and crop card
            start_time = time.time()
            cropped_image, detection_info = self._predict_card(image)
            logger.info(f'Time for crop card: {time.time() - start_time:.3f}s')
            
            if "error" in detection_info:
                return {"status": "error", "message": detection_info["error"]}

            # Validate card type and side
            detected_class = detection_info["class"]
            if required_type not in detected_class:
                return {"status": "error", "message": f"Wrong card type: required {required_type}, detected {detected_class}"}
            if required_side not in detected_class:
                return {"status": "error", "message": f"Wrong card side: required {required_side}, detected {detected_class}"}
            if "other_card" in detected_class:
                return {"status": "error", "message": "Detected card is not a valid ID card."}

            # Process image: rotate, unwarp, detect text, and extract
            # Use cached versions for better performance
            start_time = time.time()
            
            # Process rotation and unwarping in parallel
            rotate_future = self.executor.submit(self._cached_rotate_card, cropped_image)
            
            try:
                rotated_img = rotate_future.result(timeout=5.0)  # 5 second timeout
                unwarped_img = self._cached_unwarp_card(rotated_img)
                logger.info(f'Time for rotate and unwarp: {time.time() - start_time:.3f}s')
            except concurrent.futures.TimeoutError:
                logger.warning("Rotation timed out, using original image")
                unwarped_img = cropped_image
            
            # Detect text regions
            start_time = time.time()
            text_regions = self._detect_cls_text(unwarped_img, required_type)
            logger.info(f'Time for text region detection: {time.time() - start_time:.3f}s')

            # Extract text from regions (uses parallel processing internally)
            start_time = time.time()
            field_text = self._extract_text(text_regions, required_type)
            logger.info(f'Time for text extraction: {time.time() - start_time:.3f}s')

            if "error" in field_text:
                return {"status": "error", "message": field_text["error"]}

            # Map extracted text to card fields
            start_time = time.time()
            mapped_text = map_text2card(fieldname_text=field_text, require_type=required_type, require_side=required_side)
            logger.info(f'Time for map field: {time.time() - start_time:.3f}s')

            return {
                "status": "success",
                "card_info": detection_info,
                "data": mapped_text
            }

        except BadCaptureError as e:
            logger.error(f"Image quality issue: {str(e)}")
            return {"status": "error", "message": str(e)}
        except CardDetectionError as e:
            logger.error(f"Card detection issue: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error processing ID card: {str(e)}")
            return {"status": "error", "message": f"ID card processing failed: {str(e)}"}