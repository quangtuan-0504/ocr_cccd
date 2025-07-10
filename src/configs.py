from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union, Dict, List
from pymilvus import DataType, Function, FunctionType
import torch


# https://fastapi.tiangolo.com/advanced/settings/
class Settings(BaseSettings):
    """
        Configuration settings for the OCR identification card application.
    """
    # ignored if .env file is not set
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8") # extra="ignore"

    app_name: str = "OCR identification card application"


    # Box Validation
    # Ngưỡng kích thước và tỷ lệ hợp lý
    MIN_AREA_RATIO = 0.1  # Tỷ lệ diện tích tối thiểu của thẻ so với ảnh
    MAX_AREA_RATIO = 0.9  # Tỷ lệ diện tích tối đa
    ASPECT_RATIO_MIN = 1.2  # Tỷ lệ chiều dài/rộng tối thiểu
    ASPECT_RATIO_MAX = 2.0  # Tỷ lệ chiều dài/rộng tối đa

    # Server Configuration
    HOST='0.0.0.0'
    PORT=8000
    WORKERS=2

    # Logging
    LOG_LEVEL='INFO'

    # YoloObb Configuration
    YOLO_MODEL_PATH: str = 'models/yolov11n-obb.pt'
    YOLO_CONF_THRESHOLD: float = 0.8
    # Định nghĩa các lớp thẻ
    CLASS_NAMES: dict = {
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
    yoloobb_configs: dict[str, dict[str,dict[str,any]]] = {
        "initial": {
            "model_path": YOLO_MODEL_PATH
        },
        "inference": {
            "conf": YOLO_CONF_THRESHOLD,
            "iou": 0.5,
            "imgsz": (640, 640),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "rect": True,  # Use rectangular inference
            "half":	False,
            "batch": 1,	
            "max_det": 300,
            "vid_stride": 1,
            "stream_buffer": False,
            "visualize": False,
            "augment": False,
            "agnostic_nms":	False,
            "classes": None,
            "retina_masks":	False,	
            "embed": None,	
            "project":	None,
            "name":	None,
            "stream": False,	
            "verbose": False ,  
        }
    }


    # PaddleOCR Configuration
    paddleocr_configs: dict[str, dict[str,any]] = {

        "initial": {
            # Load model or not
            "doc_orientation_classify_model_name ":  None,
            "doc_orientation_classify_model_dir ":  None,

            "doc_unwarping_model_name ":  None,
            "doc_unwarping_model_dir ":  None,

            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_model_dir": None,

            "textline_orientation_model_name": None,
            "textline_orientation_model_dir": None,
            "textline_orientation_batch_size": 1,

            "text_recognition_model_name": None,
            "text_recognition_model_dir": None,
            "text_recognition_batch_size": 1,

            # Use model or not
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,

            "text_det_limit_side_len ":  100,
            "text_det_limit_type ":  None,
            "text_det_thresh ":  None,
            "text_det_box_thresh ":  None,
            "text_det_unclip_ratio ":  None,
            "text_det_input_shape ":  None,
            "text_rec_score_thresh ":  None,
            "text_rec_input_shape ":  None,
            "lang": "vi",
            "ocr_version": None,
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "enable_hpip": False
        },
        "inference": {
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": True,
            "use_textline_orientation": True
        },
    }

   
settings = Settings()

if __name__ == "__main__":
    print(settings)