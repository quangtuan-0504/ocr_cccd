import numpy as np
from typing import Tuple, List, Optional, Dict
from PIL import Image, ImageOps
import cv2
from paddleocr import DocImgOrientationClassification, TextImageUnwarping
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor

from configs import settings
from utils import (
    validate_image_quality,
    validate_bounding_box,
    crop_and_rotate_image,
    map_text2card,
    correct_image_orientation
)
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
    CLASS_CARD_NAMES = settings.CLASS_NAMES
    CLASS_FIELD_NAME = settings.FIELD_CLASS
    
    # Ngưỡng kích thước và tỷ lệ hợp lý
    MIN_AREA_RATIO = 0.1  # Tỷ lệ diện tích tối thiểu của thẻ so với ảnh
    MAX_AREA_RATIO = 0.9  # Tỷ lệ diện tích tối đa
    ASPECT_RATIO_MIN = 1.2  # Tỷ lệ chiều dài/rộng tối thiểu
    ASPECT_RATIO_MAX = 2.0  # Tỷ lệ chiều dài/rộng tối đa

    def __init__(
        self,
        cardobb_model: YOLO,
        imgorient_model: DocImgOrientationClassification,
        imgunwarp_model: TextImageUnwarping,
        textdetcls_model: dict[str, YOLO],
        rec_model: Predictor
    ):
        """
        Khởi tạo dịch vụ OCR cho thẻ căn cước/chứng minh nhân dân.

        Args:
            yolo_model (YOLO): Mô hình YOLOv11 OBB để phát hiện và cắt thẻ.
            paddleocr_model (PaddleOCR): Mô hình PaddleOCR để trích xuất văn bản.
        """
        self.cardobb_model = cardobb_model
        self.imgorient_model = imgorient_model
        self.imgunwarp_model = imgunwarp_model
        self.textdetcls_model = textdetcls_model
        self.rec_model = rec_model


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
            results = self.cardobb_model.predict(image, **settings.yolo_cardobb_configs["inference"]) # results(is list) là danh sách các kết quả dự đoán của các ảnh đầu vào
            
            if not results or not results[0].obb:
                # return None, {"error": "Không phát hiện được thẻ trong ảnh."}

                # temporal test code when model card detection is not trained
                logger.warning("Không phát hiện được thẻ trong ảnh, trả về ảnh gốc, nhớ dùng ảnh chụp sát góc để test.")
                # Trả về ảnh gốc và thông tin giả định
                return image_np, {
                    "class": "cccd_front",
                    "confidence": 0.9,
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
            
            logger.info(
                "class:", self.CLASS_CARD_NAMES[best_class]
            )
            return cropped_image, {
                "class": self.CLASS_CARD_NAMES[best_class],
                "confidence": float(best_score)
            }

        except Exception as e:
            return None, {"error": f"Lỗi khi dự đoán thẻ: {str(e)}"}

    def _rotate_card(self, cropped_image: np.ndarray) -> np.ndarray:
        """
        Rotate the input image based on predicted orientation using OpenCV.
        
        Args:
            cropped_image (np.ndarray): Input image as a numpy array.
        
        Returns:
            np.ndarray: Rotated image as a numpy array.
        """
        # Dự đoán hướng ảnh
        output_orientation_cls = self.imgorient_model.predict(input=cropped_image, **settings.paddle_imgorient_configs['inference'])
        orientation = output_orientation_cls[0]['label_names'][0] # 0, 90, 180, 270, các góc đo quay theo chiều dương(ngược kim đồng hồ), 
        # ảnh ở góc 0 độ là ảnh ntn ... hình dung đường tròn lượng giác đơn vị là ra.
        print('orientation:', orientation)
        # logger.info(f'orientation: {orientation}')
        # Khởi tạo ảnh gốc từ mảng numpy
        img_corrected = cropped_image
        
        # Xoay ảnh nếu cần
        if orientation != "0":
            # Xử lý xoay ảnh dựa trên hướng
            if orientation == "180":
                # Xoay 180 độ
                img_rotated = cv2.rotate(img_corrected, cv2.ROTATE_180)
            elif orientation == "90":
                # Xoay 90 độ ngược chiều kim đồng hồ (tương đương rotate 90 trong PIL)
                img_rotated = cv2.rotate(img_corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif orientation == "270":
                # Xoay 90 độ theo chiều kim đồng hồ (chiều âm) (tương đương rotate 270 trong PIL)
                img_rotated = cv2.rotate(img_corrected, cv2.ROTATE_90_CLOCKWISE)
        else:
            img_rotated = img_corrected
        
        return img_rotated
    
    def _unwarp_card(self, rotated_image: np.array)-> np.array:
        """
        
        """
        output_unwarp = self.imgunwarp_model.predict(
            input = rotated_image,
            **settings.paddle_imgunwarp_configs['inference']
        )

        img_unwarped = output_unwarp[0]['doctr_img']
        print(img_unwarped.shape)
        # logger.info(f'img_unwarped shape: {img_unwarped.shape}') # không chuyển về str là ko log ra được đâu => dùng f'{}' để chuyển về str
        return img_unwarped



    def _detect_cls_text(self, rotated_unwarped_img: np.array, required_type: str)-> list[dict[str,any]]:
        """
        
        """

        results = self.textdetcls_model[required_type].predict(
            source = rotated_unwarped_img,
            **settings.yolo_textdetcls_configs[required_type]['inference'] 
        )
        # print(results[0].boxes)
        fieldname_textregion_score = []
        for i,res in enumerate(results): # duyệt qua từng ảnh, only 1 img
            boxes = res.boxes
            boxes_xyxy = boxes.xyxy.cpu().numpy()  # 2d array [[],[]]
            scores = boxes.conf.cpu().numpy()  # 1d array
            classes = boxes.cls.cpu().numpy().astype(int) # 1d array
            print(f"field be detected: {list(map(lambda x: self.CLASS_FIELD_NAME[required_type][int(x)], classes))}")

            # logger.info(f"field be detected: {list(map(lambda x: self.CLASS_FIELD_NAME[int(x)], classes))}")
            # Lặp qua từng bounding box để cắt vùng ảnh
            for j, box in enumerate(boxes_xyxy):
                x_min, y_min, x_max, y_max = map(int, box)  # Chuyển đổi sang số nguyên
                # Cắt vùng ảnh chứa object
                cropped_img = rotated_unwarped_img[y_min:y_max, x_min:x_max]
                cls = classes[j]
                score = scores[j]
                fieldname_textregion_score.append(
                    {
                        'field': cls,
                        'text_region': cropped_img,
                        'score': score,
                        'y_min': y_min # dùng y_min vì mình muốn sắp theo thứ tự từ trên xuống.
                    }
                )
            
        
        return fieldname_textregion_score




    def _extract_text(self, fieldname_textregion_score: list[dict[str,any]], required_type:str) -> dict[str,dict[str,str|int|float]]:
        """

        """
        try:
            field_text = {}
            for ele in fieldname_textregion_score:
                cls_ = ele['field']
                str_cls = self.CLASS_FIELD_NAME[required_type][cls_]
                # logger.info(f"filed:{str_cls}")
                # print(str_cls)
                y_min = ele['y_min']
                score = ele['score'] # score text region det cls
                # Chuyển đổi sang định dạng PIL để đưa vào VietOCR
                cropped_img_pil = Image.fromarray(cv2.cvtColor(ele['text_region'], cv2.COLOR_BGR2RGB))
                # Nhận diện văn bản bằng VietOCR
                vietocr_text = self.rec_model.predict(cropped_img_pil)
                # print(vietocr_text)
                
                if str_cls in field_text:
                    # print("hehe")
                    if y_min > field_text[str_cls]['y_min']:
                        field_text[str_cls]['text'] = field_text[str_cls]['text'] + ' ' + vietocr_text
                    else:
                        field_text[str_cls]['text'] = vietocr_text + ' ' + field_text[str_cls]['text']
                    field_text[str_cls]['score'] = (field_text[str_cls]['score']+score)/2

                else:
                    # print("haha")
                    field_text[str_cls] = {'text':vietocr_text, 'y_min': y_min, 'score': score}
                
            return field_text
        except Exception as e:
            return {"error": f"Lỗi khi trích xuất văn bản: {str(e)}"}

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
            # image.save('analysis_error/img_origin.jpg', quality=95)
            image_np = np.array(image)
            cv2.imwrite('analysis_error/img_origin.jpg',image_np)
            # Kiểm tra ảnh đầu vào
            validate_image_quality(image=image)
            image = correct_image_orientation(image=image)
            # Dự đoán và cắt thẻ
            cropped_image, detection_info = self._predict_card(image)
            cv2.imwrite("analysis_error/img_cropped.jpg",cropped_image)
            if "error" in detection_info:
                return {"status": "error", "message": detection_info["error"]}

            # Kiểm tra loại thẻ và mặt thẻ
            detected_class = detection_info["class"]
            if required_type not in detected_class.split('_'):
                return {"status": "error", "message": f"Loại thẻ không đúng: yêu cầu {required_type}, phát hiện {detected_class}"}
            if required_side not in detected_class.split('_'):
                return {"status": "error", "message": f"Mặt thẻ không đúng: yêu cầu {required_side}, phát hiện {detected_class}"}
            if "other_card" in detected_class:
                return {"status": "error", "message": "Phát hiện thẻ không thuộc loại căn cước hoặc căn cước công dân."}
            
            rotated_img = self._rotate_card(cropped_image=cropped_image)
            unwarped_img = self._unwarp_card(rotated_image=rotated_img)
            cv2.imwrite("analysis_error/img_cropped_rotated_unwarped.jpg",unwarped_img)
            text_regions = self._detect_cls_text(rotated_unwarped_img=unwarped_img, required_type = required_type)
            # Trích xuất văn bản
            field_text = self._extract_text(text_regions, required_type= required_type)
            if "error" in field_text:
                return {"status": "error", "message": field_text["error"]}

            # map field của từng loại thẻ
            map_field_text = map_text2card(
                fieldname_text=field_text,
                require_type=required_type,
                require_side=required_side
            )
            
            return {
                "status": "success",
                "card_info": detection_info,
                "data": map_field_text
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