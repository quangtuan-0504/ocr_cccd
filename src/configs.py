from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Any, Union
import torch


class Settings(BaseSettings):
    """
    Configuration settings for the OCR identification card application.
    """
    # Configuration for .env file loading
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application settings
    app_name: str = "OCR identification card application"

    # Card validation thresholds
    MIN_AREA_RATIO: float = 0.1     # Minimum card area ratio relative to image
    MAX_AREA_RATIO: float = 0.95    # Maximum card area ratio relative to image
    ASPECT_RATIO_MIN: float = 1.0   # Minimum card aspect ratio (width/height)
    ASPECT_RATIO_MAX: float = 2.0   # Maximum card aspect ratio (width/height)

    # Card type class names
    CLASS_NAMES: Dict[int, str] = {
        0: "cccd_front",
        1: "cccd_back",
        2: "cc_front",
        3: "cc_back",
        4: "other_card"
    }

    # YOLOv11 OBB model for card detection
    YOLO_MODEL_PATH: str = "model/det_card/best.pt"
    YOLO_CONF_THRESHOLD: float = 0.5
    
    yolo_cardobb_configs: Dict[str, Dict[str, Any]] = {
        "initial": {
            "model": YOLO_MODEL_PATH
        },
        "inference": {
            "conf": YOLO_CONF_THRESHOLD,
            "iou": 0.3,
            "imgsz": (640, 640),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "rect": True,
            "half": False,
            "batch": 1,
            "max_det": 10,
            "vid_stride": 1,
            "stream_buffer": False,
            "visualize": False,
            "augment": False,
            "agnostic_nms": False,
            "classes": None,
            "retina_masks": False,
            "embed": None,
            "project": None,
            "name": None,
            "stream": False,
            "verbose": False,
        }
    }

    # PaddleOCR image orientation model
    paddle_imgorient_configs: Dict[str, Dict[str, Any]] = {
        "initial": {
            "model_name": "PP-LCNet_x1_0_doc_ori",
            "model_dir": "model/PP-LCNet_x1_0_doc_ori",
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "enable_hpi": False,
        },
        "inference": {
            "batch_size": 1
        }
    }

    # PaddleOCR image unwarping model
    paddle_imgunwarp_configs: Dict[str, Dict[str, Any]] = {
        "initial": {
            "model_name": "UVDoc",
            "model_dir": "model/UVDoc",
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "enable_hpi": False,
        },
        "inference": {
            "batch_size": 1
        }
    }

    # Field class definitions for different card types
    FIELD_CLASS: Dict[str, Dict[int, str]] = {
        # Citizen card (cc) field classes
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
        # Citizen ID card (cccd) field classes
        'cccd': {
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

    # YOLO text detection and classification models
    yolo_textdetcls_configs: Dict[str, Dict[str, Any]] = {
        # Citizen card (cc) text detection model
        "cc": {
            "initial": {
                "model": "model/textdet/cc/best.pt"
            },
            "inference": {
                "conf": YOLO_CONF_THRESHOLD,
                "iou": 0.3,
                "imgsz": (640, 640),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "rect": True,
                "half": False,
                "batch": 1,
                "max_det": 10,
                "vid_stride": 1,
                "stream_buffer": False,
                "visualize": False,
                "augment": False,
                "agnostic_nms": False,
                "classes": None,
                "retina_masks": False,
                "embed": None,
                "project": 'analysis_error',
                "name": 'cc_det',
                "stream": False,
                "verbose": False,
                "save": True
            }
        },
        # Citizen ID card (cccd) text detection model
        "cccd": {
            "initial": {
                "model": "model/textdet/cccd/best.pt"
            },
            "inference": {
                "conf": YOLO_CONF_THRESHOLD,
                "iou": 0.3,
                "imgsz": (640, 640),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "rect": True,
                "half": False,
                "batch": 1,
                "max_det": 10,
                "vid_stride": 1,
                "stream_buffer": False,
                "visualize": False,
                "augment": False,
                "agnostic_nms": False,
                "classes": None,
                "retina_masks": False,
                "embed": None,
                "project": 'analysis_error',
                "name": 'cccd_det',
                "stream": False,
                "verbose": False,
                "save": True
            }
        }
    }

    # VietOCR configuration
    vietocr_configs: Dict[str, str] = {
        "weights": "model/vietocr/transformerocr.pth",
        "config": "model/vietocr/config.yml",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }


# Create settings instance
settings = Settings()

if __name__ == "__main__":
    print(settings)