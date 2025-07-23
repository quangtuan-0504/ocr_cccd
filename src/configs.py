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

    # Box Validation
    # Ngưỡng kích thước và tỷ lệ hợp lý
    MIN_AREA_RATIO: float= 0.1  # Tỷ lệ diện tích tối thiểu của thẻ so với ảnh
    MAX_AREA_RATIO: float = 0.95  # Tỷ lệ diện tích tối đa
    ASPECT_RATIO_MIN: float = 1  # Tỷ lệ chiều dài/rộng tối thiểu
    ASPECT_RATIO_MAX: float = 2.0  # Tỷ lệ chiều dài/rộng tối đa

    # YoloObb Configuration
    YOLO_MODEL_PATH: str = "model/det_card/best.pt"
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
            "model_dir": "model/PP-LCNet_x1_0_doc_ori",
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
            "model_dir": "model/UVDoc",
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
            1: 'name',
            2: 'date_of_birth',
            3: 'sex',
            4: 'nation',
            5: 'place_of_residence',
            6: 'place_of_birth',
            7: 'date_of_issue',
            8: 'date_of_expiry',
            9: 'face',
            10: 'qr'
        },
        'cccd': {
            # 0: 'id',
            # 1: 'name',
            # 2: 'date_of_birth',
            # 3: 'sex',
            # 4: 'nation',
            # 5: 'place_of_origin',
            # 6: 'place_of_residence',
            # 7: 'date_of_expiry',
            # 8: 'personal_iden',
            # 9: 'date_of_issue',
            # 10: 'face',
            # 11: 'qr'

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
                "model": "model/textdet/cc/best.pt"
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
                "project":	'analysis_error',
                "name":	'cc_det',
                "stream": False,	
                "verbose": False ,  
                "save": True
            }
        },
        "cccd": {
            "initial": {
                "model": "model/textdet/cccd/best.pt"
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
                "project":	'analysis_error',
                "name":	'cccd_det',
                "stream": False,	
                "verbose": False ,  
                "save": True
            }
        }
    }

    vietocr_configs: dict[str,str] = {
        'model_path': 'model/vietocr/vgg_seq2seq.pth',
        'base_cf': 'model/vietocr/base.yml',
        'cf': 'model/vietocr/vgg-seq2seq.yml'
    }
   
settings = Settings()

if __name__ == "__main__":
    print(settings)