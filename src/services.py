import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from PIL import Image
import cv2
from paddleocr import DocImgOrientationClassification, TextImageUnwarping
from ultralytics import YOLO
import time
import concurrent.futures
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from vietocrOnnxModule.tool.translate import translate_onnx, process_input
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
import torch

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
        vietocr_onnx: Tuple
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
        self.rec_model = vietocr_onnx[0]
        self.vocab = vietocr_onnx[1]
        self.vietocr_cf = vietocr_onnx[2]
        
        # Initialize thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
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
            
            # Run YOLOv11 inference with no_grad to save memory
            with torch.no_grad():
                results = self.cardobb_model.predict(image, **inference_config)

            if not results or not results[0].obb:
                # Clean up before returning
                del image_np, results
                return None, {"error": "No card detected in image"}

            obb = results[0].obb
            # Move tensors to CPU immediately and convert to numpy
            boxes = obb.xywhr.cpu().numpy()
            scores = obb.conf.cpu().numpy()
            classes = obb.cls.cpu().numpy().astype(int)

            # Find the detection with highest confidence
            max_conf_idx = scores.argmax()
            best_box = boxes[max_conf_idx]
            best_class = classes[max_conf_idx]
            best_score = scores[max_conf_idx]

            img_height, img_width = image_np.shape[:2]
            w_ratio, h_ratio = validate_bounding_box(bbox=best_box, image_size=(img_width, img_height))
            cropped_image = crop_and_rotate_image(image_np, best_box)

            # Clean up tensors and variables immediately
            del obb, boxes, scores, classes
            del image_np, results

            logger.info(f"Detected card class: {self.CLASS_CARD_NAMES[best_class]} with confidence {best_score:.3f}")
            return cropped_image, {
                "class": self.CLASS_CARD_NAMES[best_class],
                "confidence": float(best_score),
                "wh_ratio": (w_ratio,h_ratio)
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
        output = self.imgorient_model.predict(
            input=cropped_image, 
            **settings.paddle_imgorient_configs['inference']
        )
        orientation = output[0]['label_names'][0]
        logger.info(f"Predicted orientation: {orientation}")
        del output

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
        del output
        return unwarped_img

    def _detect_cls_text(self, rotated_unwarped_img: np.ndarray, required_type: str) -> List[Dict[str, Any]]:
        """Detect and classify text regions in the unwarped image.

        Args:
            rotated_unwarped_img: Unwarped image as a numpy array.
            required_type: Type of card ("cccd" or "cc").

        Returns:
            List of dictionaries containing field, text region, score, and y_min.
        """
        # Use optimized inference settings
        inference_config = settings.yolo_textdetcls_configs[required_type]['inference'].copy()
        inference_config["verbose"] = False
        
        if inference_config["save"]:
            img = Image.fromarray(rotated_unwarped_img)
        else:
            img = rotated_unwarped_img
        
        
        fieldname_textregion_score = []
        
        try:
            # Run YOLOv11 inference with no_grad to save memory
            with torch.no_grad():
                results = self.textdetcls_model[required_type].predict(
                    source=img, **inference_config
                )

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                    
                # Move tensors to CPU immediately and convert to numpy
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
                    str_cls = self.CLASS_FIELD_NAME[required_type][classes[j]]
                    
                    fieldname_textregion_score.append({
                        'field': str_cls,
                        'text_region': cropped_img, # np.array
                        'score': scores[j],
                        'y_min': y_min
                    })
                
                # Clean up tensors for this result
                del boxes, boxes_xyxy, scores, classes

        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
        finally:
            # Clean up all remaining tensors and variables
            del img, rotated_unwarped_img, results
            
        return fieldname_textregion_score

    def _extract_text_region(self, region_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from a single region using VietOCR.
        
        Args:
            region_data: Dictionary containing field data and image region.
            required_type: Type of card ("cccd" or "cc").
            
        Returns:
            Dictionary with field name, extracted text and metadata.
        """
        str_cls = region_data['field']
        y_min = region_data['y_min']
        score = region_data['score']
        
        try:
            # Convert image for OCR
            cropped_img = cv2.cvtColor(region_data['text_region'], cv2.COLOR_BGR2RGB)
            # Process input image - returns numpy array ready for ONNX inference
            img = process_input(cropped_img, self.vietocr_cf['dataset']['image_height'], 
                            self.vietocr_cf['dataset']['image_min_width'], self.vietocr_cf['dataset']['image_max_width'])  
            
            # Process with ONNX model - no need for additional np.array conversion
            vietocr_text = translate_onnx(img, self.rec_model)[0].tolist()
            vietocr_text = self.vocab.decode(vietocr_text)
            
            # Clean up variables to prevent memory leaks
            del cropped_img, img
            
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

    def _extract_text(self, fieldname_textregion_score: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
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
            
            face_qr_img = {}
            # Use parallel processing for text extraction
            futures = []
            for region_data in fieldname_textregion_score:
                if region_data['field'] not in ['face,qr']:
                    futures.append(
                        self.executor.submit(
                            self._extract_text_region, 
                            region_data
                        )
                    )
                else:
                    face_qr_img[region_data['field']] = region_data['text_region']
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

            return field_text, face_qr_img 
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {"error": f"Text extraction failed: {str(e)}"}

    def process_id_card(self, image: Image.Image) -> Dict[str, Any]:
        """Process an ID card image and extract information.

        Args:
            image: Input image in PIL format.
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
            # cv2.imwrite('analysis_error/img_cropped.jpg', cropped_image)
            logger.info(f'Time for crop card: {time.time() - start_time:.3f}s')
            
            if "error" in detection_info:
                return {
                    "errorCode": "2",
                    "errorMessage":detection_info["error"],
                    "data": []
                }

            # Validate card type and side
            detected_class = detection_info["class"]
            if "other_card" in detected_class:
                return {
                    "errorCode": "2",
                    "errorMessage": "Detected card is not a valid ID card.",
                    "data": []
                }

            # Process image: rotate, unwarp, detect text, and extract
            start_time = time.time()
            rotated_img = self._rotate_card(cropped_image)
            
            w_ratio = detection_info['wh_ratio'][0]
            h_ratio = detection_info['wh_ratio'][1]
            if w_ratio < 0.9 and h_ratio < 0.9:
                unwarped_img = self._unwarp_card(rotated_img)
            else:
                unwarped_img = rotated_img
            logger.info(f'Time for rotate and unwarp: {time.time() - start_time:.3f}s')
            
            card_type, card_side = detected_class.split('_')
            # Detect text regions
            start_time = time.time()
            regions = self._detect_cls_text(
                rotated_unwarped_img=unwarped_img,
                required_type = card_type
            )
            logger.info(f'Time for text region detection: {time.time() - start_time:.3f}s')

            # Extract text from regions (uses parallel processing internally)
            start_time = time.time()
            field_text, face_qr = self._extract_text(
                fieldname_textregion_score = regions
            )
            logger.info(f'Time for text extraction: {time.time() - start_time:.3f}s')

            if "error" in field_text:
                return {
                    "errorCode": "3",
                    "errorMessage": field_text["error"],
                    "data": []
                }

            # Map extracted text to card fields
            start_time = time.time()
            mapped_text = map_text2card(fieldname_text=field_text, require_type=card_type, require_side=card_side)
            logger.info(f'Time for map field: {time.time() - start_time:.3f}s')

            return {
                "errorCode": "0",
                "errorMessage": "no error",
                "data": [mapped_text]
            }

        except BadCaptureError as e:
            logger.error(f"Image quality issue: {str(e)}")
            # return {"status": "error", "message": str(e)}
            return {
                "errorCode": "1",
                "errorMessage": str(e),
                "data": []
            }
        except CardDetectionError as e:
            logger.error(f"Card detection issue: {str(e)}")
            # return {"status": "error", "message": str(e)}
            return {
                "errorCode": "2",
                "errorMessage": str(e),
                "data": []
            }
        except Exception as e:
            logger.error(f"Error processing ID card: {str(e)}")
            # return {"status": "error", "message": str(e)}
            return {
                "errorCode": "1",
                "errorMessage": f"ID card processing failed: {str(e)}",
                "data": []
            }
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()