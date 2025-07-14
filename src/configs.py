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
        0: "cccd_front",
        1: "cccd_back",
        2: "cc_front",
        3: "cc_back",
        4: "other_card"
    }
    yolo_cardobb_configs: dict[str, dict[str,dict[str,any]]] = {
        "initial": {
            "model_path": YOLO_MODEL_PATH
        },
        "inference": {
            "conf": YOLO_CONF_THRESHOLD,
            "iou": 0.4,
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

    paddle_imgorient_configs: dict[str,dict[str,any]] = {
        "initial": {
            "model_name	": "PP-LCNet_x1_0_doc_ori",
            "model_dir": "",
            "device": "gpu",
            "enable_hpi": False,
        },
        "inference": {
           "batch_size": 1
        }
    }

    paddle_imgunwarp_configs: dict[str,dict[str,any]] = {
        "initial": {
            "model_name	": "UVDoc",
            "model_dir": "",
            "device": "gpu",
            "enable_hpi": False,

        },
        "inference": {
           "batch_size": 1
        }
    }
    FIELD_CLASS: dict[int,str] = {
        0: 'id number',
        1: 'full name',
        2: 'date of birth',
        3: 'sex',
        4: 'nationality',
        6: 'place of origin',
        7: 'place of residence',
        8: 'date of expiry'
    }
    yolo_textdetcls_configs: dict[str,dict[str,any]] = {
        "initial": {
            "model_path": "yolo11n.pt"
        },
        "inference": {
            "conf": YOLO_CONF_THRESHOLD,
            "iou": 0.4,
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


   
settings = Settings()

if __name__ == "__main__":
    print(settings)