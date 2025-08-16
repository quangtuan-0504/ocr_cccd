import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from PIL import Image
import cv2
from paddleocr import TextRecognition
import time
import concurrent.futures
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from yoloOnnxModule.prediction import YOLOv11OnnxRuntime
from vietocrOnnxModule.tool.translate import translate_onnx, process_input
from configs import settings
from utils import (
    validate_image_quality,
    validate_and_crop_card,
    map_text2card,
    correct_image_orientation
)


from logger import logger
from exceptions import CardDetectionError, BadCaptureError
import onnxruntime

class IdCardOCRService:
    """Service for processing ID card images using YOLOv11 OBB, PaddleOCR and VietOCR."""
    
    # Card class definitions
    CLASS_CARD_NAMES = settings.CLASS_NAMES
    CLASS_FIELD_NAME = settings.FIELD_CLASS
    
    def __init__(
        self,
        yolo_4cornercarddet_model: YOLOv11OnnxRuntime,
        yolo_textregiondet_model: Dict[str, YOLOv11OnnxRuntime],
        paddle_imgrec_model: TextRecognition,
        vietocr_rec_model: Tuple
    ):
        """Initialize the ID card OCR service.

        Args:
            yolo_4cornercarddet_model: YOLOv11 for detection corner and card.
            yolo_textregiondet_model: Dictionary of YOLO models for text region detection.
            paddle_imgrec_model: PaddleOCR model for text recognition.
            vietocr_rec_model: VietOCR model for text recognition.
        """
        self.cornercarddet_model = yolo_4cornercarddet_model
        self.textregiondet_model = yolo_textregiondet_model
        self.mrzrec_model = paddle_imgrec_model
        self.rec_model = vietocr_rec_model[0]
        self.vocab = vietocr_rec_model[1]
        self.vietocr_cf = vietocr_rec_model[2]

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
            inference_config = settings.yolo_4cornercarddet_configs["inference"]
            
            # Run YOLOv11 inference
            results = self.cornercarddet_model.predict(image_np, inference_config)
            # Access results
            boxes_xyxy = results['boxes_xyxy_original']  # [x1, y1, x2, y2] format
            boxes_xywh = results['boxes_xywh_original']  # [x, y, w, h] format
            scores = results['scores']
            class_ids = results['class_ids']
            # Pass in the original image and boxes corresponding it to crop the card on the original image.
            cropped_image, card_type_name, card_type_conf = validate_and_crop_card(boxes_xywh, scores, class_ids, image_np)
            # Clean up tensors and variables immediately
            del image_np, boxes_xywh, scores, class_ids, boxes_xyxy
            return cropped_image, {
                "class": card_type_name,
                "confidence": float(card_type_conf)
            }

        except Exception as e:
            logger.error(f"Error predicting card: {str(e)}")
            
            return None, {"error": f"Card prediction failed: {str(e)}"}

    def _detect_textregion(self, cropped_image: np.ndarray, required_type: str) -> List[Dict[str, Any]]:
        """Detect and classify text regions in the cropped image.

        Args:
            cropped_image: Cropped image as a numpy array.
            required_type: Type of card ("cccd" or "cc").

        Returns:
            List of dictionaries containing field, text region, score, and y_min.
        """
        # Use optimized inference settings
        inference_config = settings.yolo_textregiondet_configs[required_type]['inference']
        
        fieldname_textregion_score = []
        
        try:
            # Run YOLOv11 inference
            results = self.textregiondet_model[required_type].predict(cropped_image, inference_config)

            boxes_xyxy = results['boxes_xyxy_original']
            boxes_xywh = results['boxes_xywh_original']
            scores = results['scores']
            classes = results['class_ids']

            detected_fields = [self.CLASS_FIELD_NAME[required_type][int(cls)] for cls in classes]
            logger.info(f"Detected fields: {detected_fields}")

            # Process all detected regions at once
            for j, box in enumerate(boxes_xyxy):
                x_min, y_min, x_max, y_max = map(int, box)
                # Ensure coordinates are within bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(cropped_image.shape[1], x_max)
                y_max = min(cropped_image.shape[0], y_max)
                
                # Skip invalid regions
                if x_min >= x_max or y_min >= y_max:
                    logger.warning(f"Skipping invalid text region: {x_min},{y_min},{x_max},{y_max}")
                    continue
                    
                textregion_img = cropped_image[y_min:y_max, x_min:x_max]
                str_cls = self.CLASS_FIELD_NAME[required_type][classes[j]]
                
                fieldname_textregion_score.append({
                    'field': str_cls,
                    'text_region': textregion_img, # np.array
                    'score': scores[j],
                    'y_min': y_min
                })
            
            # Clean up tensors for this result
            del boxes_xyxy, boxes_xywh, scores, classes, cropped_image

        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
        
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
    def _extract_mrz(self, region_data: Dict[str, Any]) -> Dict[str, Any]:
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
            # For MRZ, use PaddleOCR recognition model
            output = self.mrzrec_model.predict(
                input=region_data['text_region'],
                **settings.paddle_imgrec_configs['inference']
            )
            # print(output) **
            mrz_text = output[0]['rec_text']
            del output
            logger.info(f"Extracted text for field {str_cls}: {mrz_text}")
           
            return {
                'field_name': str_cls,
                'text': mrz_text,
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
            mrzs = []
            for region_data in fieldname_textregion_score:
                if region_data['field'] not in ['face', 'qr', 'mrz']:
                    futures.append(
                        self.executor.submit(
                            self._extract_text_region,
                            region_data
                        )
                    )
                elif region_data['field'] == 'mrz':
                    mrzs.append(region_data)
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
            for region_data in sorted(mrzs, key=lambda x: x['y_min']):
                result = self._extract_mrz(region_data=region_data)
                if 'error' in result:
                    logger.warning(f"Text extraction issue: {result.get('error', 'Unknown error')}")
                    continue
                   
                str_cls = result['field_name']
                text = result['text']
                y_min = result['y_min']
                score = result['score']
                if str_cls in field_text:
                    field_text[str_cls]['text'].append(text)
                    field_text[str_cls]['score'] = (field_text[str_cls]['score'] + score) / 2
                else:
                    field_text[str_cls] = {'text': [text], 'y_min': y_min, 'score': score}
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
            
            detected_class = detection_info["class"]
            card_type, card_side = detected_class.split('_')
            # Detect text regions
            start_time = time.time()
            regions = self._detect_textregion(
                cropped_image=cropped_image,
                required_type=card_type
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
        # finally:
        #     if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        #         pass