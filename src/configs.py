from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union, Dict, List
import torch


# https://fastapi.tiangolo.com/advanced/settings/
class Settings(BaseSettings):
    """
        Configuration settings for the OCR identification card application.
    """
    # ignored if .env file is not set
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8") # extra="ignore"

    app_name: str = "OCR identification card application"

    # YoloObb Configuration
    YOLO_MODEL_PATH: str = "./model/yolo11n-obb.pt"
    YOLO_CONF_THRESHOLD: float = 0.5
    # Định nghĩa các lớp thẻ
    CLASS_NAMES: dict = {
        0: "cccd_front",
        1: "cccd_back",
        2: "cc_front",
        3: "cc_back",
        4: "other_card"
    }
    yolo_cardobb_configs: dict[str, dict[str,any]] = {
        "initial": {
            "model": YOLO_MODEL_PATH # nó vào đường dẫn này tìm model, nếu không có thì sẽ tải trên mạng về lưu vô đường dẫn này, rồi từ đây load lên chạy.
        },
        "inference": {
            "conf": YOLO_CONF_THRESHOLD,
            "iou": 0.3,
            "imgsz": (640, 640),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "rect": True,  # Use rectangular inference
            "half":	False,
            "batch": 1,	
            "max_det": 10,
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
            "model_name": "PP-LCNet_x1_0_doc_ori",
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
            "model_name": "UVDoc",
            "model_dir": "",
            "device": "gpu",
            "enable_hpi": False,

        },
        "inference": {
           "batch_size": 1
        }
    }
    FIELD_CLASS: dict[str,dict[int,str]] = {
        # Real data
        'cc': {
            0: 'id',
            2: 'name',
            3: 'date_of_birth',
            4: 'sex',
            5: 'nation',
            6: 'place_of_residence',
            7: 'place_of_birth',
            8: 'date_of_issue',
            9: 'date_of_expiry'
        },
        'cccd': {
            # 0: 'id',
            # 2: 'name',
            # 3: 'date_of_birth',
            # 4: 'sex',
            # 5: 'nation',
            # 6: 'place_of_origin',
            # 7: 'place_of_residence',
            # 8: 'date_of_expiry',
            # 9: 'personal_iden',
            # 10: 'date_of_issue'

            # toy data
            0: 'date_of_birth',
            1: 'date_of_expiry',
            2: 'gender',
            3: 'hometown',
            4: 'id',
            5: 'name',
            6: 'nation',
            7: 'permanent_residence'
        }
    }
    yolo_textdetcls_configs: dict[str,dict[str,any]] = {
        "cc": {
            "initial": {
                "model": "/workspace/ocr_cccd/model/best.pt"
            },
            "inference": {
                "conf": YOLO_CONF_THRESHOLD,
                "iou": 0.3,
                "imgsz": (640, 640),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "rect": True,  # Use rectangular inference
                "half":	False,
                "batch": 1,	
                "max_det": 10,
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
        },
        "cccd": {
            "initial": {
                "model": "/workspace/ocr_cccd/model/best.pt"
            },
            "inference": {
                "conf": YOLO_CONF_THRESHOLD,
                "iou": 0.3,
                "imgsz": (640, 640),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "rect": True,  # Use rectangular inference
                "half":	False,
                "batch": 1,	
                "max_det": 10,
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
    }

   
settings = Settings()

if __name__ == "__main__":
    print(settings)