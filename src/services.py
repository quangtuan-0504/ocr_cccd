import numpy as np
from typing import Tuple, List, Optional, Dict
from PIL import Image
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO

from configs import settings
from utils import validate_image_quality, validate_bounding_box, crop_and_rotate_image
from exceptions import (
    BadCaptureError,
    CardDetectionError,
    BadBoxError,
    ModelNotLoadedError
    )
from logger import logger

class IdCardOCRService:
    """
        Dịch vụ chính để xử lý ảnh thẻ căn cước/chứng minh nhân dân sử dụng YOLOv11 OBB và PaddleOCR.
        Reference:
        https://docs.ultralytics.com/modes/predict/#inference-sources
        https://docs.ultralytics.com/modes/predict/#inference-arguments
        https://docs.ultralytics.com/vi/modes/predict/#obb
        https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.OBB
        https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html

    
    """
    
    # Định nghĩa các lớp thẻ
    CLASS_NAMES = {
        0: "cccd_front_correct",
        1: "cccd_front_inverted",
        2: "cccd_back_correct",
        3: "cccd_back_inverted",
        4: "cc_front_correct",
        5: "cc_front_inverted",
        6: "cc_back_correct",
        7: "cc_back_inverted",
        8: "other_card"
    }
    
    # Ngưỡng kích thước và tỷ lệ hợp lý
    MIN_AREA_RATIO = 0.1  # Tỷ lệ diện tích tối thiểu của thẻ so với ảnh
    MAX_AREA_RATIO = 0.9  # Tỷ lệ diện tích tối đa
    ASPECT_RATIO_MIN = 1.2  # Tỷ lệ chiều dài/rộng tối thiểu
    ASPECT_RATIO_MAX = 2.0  # Tỷ lệ chiều dài/rộng tối đa

    def __init__(
        self,
        yolo_model: YOLO,
        paddleocr_model: PaddleOCR
    ):
        """
        Khởi tạo dịch vụ OCR cho thẻ căn cước/chứng minh nhân dân.

        Args:
            yolo_model (YOLO): Mô hình YOLOv11 OBB để phát hiện và cắt thẻ.
            paddleocr_model (PaddleOCR): Mô hình PaddleOCR để trích xuất văn bản.
        """
        self.yolo_model = yolo_model
        self.paddleocr_model = paddleocr_model


    def _predict_card(self, image: Image.Image) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Dự đoán và phát hiện thẻ trong ảnh sử dụng YOLOv11 OBB.

        Args:
            image (Image.Image): Ảnh đầu vào định dạng PIL.

        Returns:
            Tuple[Optional[np.ndarray], Optional[Dict]]: Ảnh đã cắt (nếu có) và thông tin dự đoán.
        """
        try:
            # Chuyển ảnh PIL sang numpy
            image_np = np.array(image)
            # Dự đoán với YOLOv11 OBB
            results = self.yolo_model.predict(image, **settings.yoloobb_configs["inference"]) # results(is list) là danh sách các kết quả dự đoán của các ảnh đầu vào
            
            if not results or not results[0].obb:
                # return None, {"error": "Không phát hiện được thẻ trong ảnh."}

                # temporal test code when model card detection is not trained
                logger.warning("Không phát hiện được thẻ trong ảnh, trả về ảnh gốc.")
                # Trả về ảnh gốc và thông tin giả định
                return image_np, {
                    "class": "cccd_front_correct",
                    "confidence": 0.9,
                    "is_inverted": False
                }   
            
            # Lấy kết quả OBB
            obb = results[0].obb # OBB object
            boxes = obb.xywhr.cpu().numpy()  # Hộp giới hạn OBB theo định dạng xywhr
            scores = obb.conf.cpu().numpy()  # Độ tin cậy
            classes = obb.cls.cpu().numpy().astype(int)  # Lớp

            # Chọn hộp có độ tin cậy cao nhất
            max_conf_idx = scores.argmax()
            best_box = boxes[max_conf_idx]
            best_class = classes[max_conf_idx] # int
            best_score = scores[max_conf_idx]
            
            # Lấy kích thước ảnh
            img_height, img_width = image_np.shape[:2]
            # Kiểm tra tính hợp lệ của bounding box
            validate_bounding_box(
                bbox=best_box,
                image_size=(img_width, img_height)
            )
            # Cắt ảnh theo hộp OBB
            cropped_image = crop_and_rotate_image(image_np, best_box)
            
            # Xoay ảnh nếu thẻ bị ngược
            is_inverted = "inverted" in self.CLASS_NAMES[best_class]
            if is_inverted:
                cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_180)

            return cropped_image, {
                "class": self.CLASS_NAMES[best_class],
                "confidence": float(best_score),
                "is_inverted": is_inverted
            }

        except Exception as e:
            return None, {"error": f"Lỗi khi dự đoán thẻ: {str(e)}"}

    def _extract_text(self, cropped_image: np.ndarray) -> List[Dict]:
        """
        Trích xuất văn bản từ ảnh đã cắt sử dụng PaddleOCR.

        Args:
            cropped_image (np.ndarray): Ảnh đã cắt chứa thẻ.

        Returns:
            List[Dict]: Danh sách các văn bản được trích xuất với tọa độ và độ tin cậy.
        """
        try:
            result = self.paddleocr_model.predict(
                input=cropped_image,
                **settings.paddleocr_configs["inference"]    
            )
            extracted_text = []
            for res_per_img in result: # Only one image in result
                rec_texts = res_per_img["rec_texts"]  # Kết quả nhận dạng văn bản
                rec_scores = res_per_img["rec_scores"]  # Độ tin cậy của văn bản
                rec_boxes = res_per_img["rec_boxes"]  # Tọa độ bounding box của văn bản

                extracted_text.append({
                    "texts": rec_texts,
                    "boxs": rec_boxes,
                    "confidences": rec_scores
                })
            return extracted_text
        except Exception as e:
            return [{"error": f"Lỗi khi trích xuất văn bản: {str(e)}"}]

    def process_id_card(
        self,
        image: Image.Image,
        required_type: str = "cccd",
        required_side: str = "front"
    ) -> Dict:
        """
        Xử lý ảnh thẻ căn cước/chứng minh nhân dân và trích xuất thông tin.

        Args:
            image (Image.Image): Ảnh đầu vào định dạng PIL.
            required_type (str): Loại thẻ yêu cầu ("cccd" hoặc "cc").
            required_side (str): Mặt thẻ yêu cầu ("front" hoặc "back").

        Returns:
            Dict: Kết quả chứa thông tin thẻ và văn bản trích xuất, hoặc thông báo lỗi.
        """
        try:
            # Kiểm tra ảnh đầu vào
            validate_image_quality(image=image)

            # Dự đoán và cắt thẻ
            cropped_image, detection_info = self._predict_card(image)
            
            if "error" in detection_info:
                return {"status": "error", "message": detection_info["error"]}

            # Kiểm tra loại thẻ và mặt thẻ
            detected_class = detection_info["class"]
            if required_type not in detected_class:
                return {"status": "error", "message": f"Loại thẻ không đúng: yêu cầu {required_type}, phát hiện {detected_class}"}
            if required_side not in detected_class:
                return {"status": "error", "message": f"Mặt thẻ không đúng: yêu cầu {required_side}, phát hiện {detected_class}"}
            if "other_card" in detected_class:
                return {"status": "error", "message": "Phát hiện thẻ không thuộc loại căn cước hoặc căn cước công dân."}

            # Trích xuất văn bản
            extracted_text = self._extract_text(cropped_image)

            return {
                "status": "success",
                "card_info": detection_info,
                "extracted_text": extracted_text
            }

        except Exception as e:
            return {"status": "error", "message": f"Lỗi xử lý ảnh thẻ: {str(e)}"}

    def __call__(self, *args, **kwargs) -> Dict:
        """
        Gọi nhanh phương thức process_id_card.

        Args:
            *args: Tham số vị trí.
            **kwargs: Tham số từ khóa.

        Returns:
            Dict: Kết quả từ process_id_card.
        """
        return self.process_id_card(*args, **kwargs)