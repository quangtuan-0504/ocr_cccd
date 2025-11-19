from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Any, Union
import onnxruntime as ort
import yaml
import os
from pathlib import Path

class Settings(BaseSettings):
    """
    Configuration settings for the OCR identification card application.
    """
    # Configuration for .env file loading
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @staticmethod
    def _load_yaml_config():
        """Load configuration from YAML file"""
        yaml_path = Path(__file__).parent.parent / "configs.yml"
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        else:
            raise ValueError("not found file configs path to models")

    config: Dict[str,Dict[str,str|float|int]] = _load_yaml_config()  

    # Application settings
    app_name: str = "OCR identification card application"

    # Card validation thresholds
    MIN_AREA_RATIO: float = 0.1     # Minimum card area ratio relative to image
    MAX_AREA_RATIO: float = 0.95    # Maximum card area ratio relative to image
    ASPECT_RATIO_MIN: float = 1.0   # Minimum card aspect ratio (width/height)
    ASPECT_RATIO_MAX: float = 2.0   # Maximum card aspect ratio (width/height)

    # Card type class names
    CLASS_NAMES: Dict[int, str] = {
        0: 'top_left',
        1: 'top_right',
        2: 'bottom_right',
        3: 'bottom_left',
        4: 'cc_front',
        5: 'cc_back',
        6: "cccd_front",
        7: "cccd_back",
        8: "other_card"

        # toy model class names
        # 0: 'bottom_left',
        # 1: 'bottom_right',
        # 2: 'cc_back',
        # 3: 'cc_front',
        # 4: 'top_left',
        # 5: 'top_right',
        # 6: "cccd_front",
        # 7: "cccd_back",
        # 8: "other_card"
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
            10: 'qr',
            11: 'mrz'
        },
        # Citizen ID card (cccd) field classes
        'cccd': {
            0: 'id',
            1: 'name',
            2: 'date_of_birth',
            3: 'sex',
            4: 'nation',
            5: 'place_of_origin',
            6: 'place_of_residence',
            7: 'date_of_expiry',
            8: 'personal_iden',
            9: 'date_of_issue',
            10: 'face',
            11: 'qr',
            12: 'mrz'
        }
    }

    YOLO_DET_4CORNERCARD_MODEL_PATH: str = config['model_paths']['yolo_det_4cornercard_model_path']
    YOLO_DET_TEXTCC_MODEL_PATH: str = config['model_paths']['yolo_det_textcc_model_path']
    YOLO_DET_TEXTCCCD_MODEL_PATH: str = config['model_paths']['yolo_det_textcccd_model_path']

    VIETOCR_REC_TEXTREGION_MODEL_PATH: str = config['model_paths']['vietocr_rec_textregion_model_path']

    YOLO_CONF_THRESHOLD_DET_CORNERCARD: float = 0.2
    YOLO_CONF_THRESHOLD_DET_TEXT: float = 0.5


    PROVIDERS_ONNXRUNTIME: List = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    # YOLOv11 detection 4 corner card model for card detection
    yolo_4cornercarddet_configs: Dict[str, Dict[str, Any]] = {
        "initial": {
            "path_or_bytes": YOLO_DET_4CORNERCARD_MODEL_PATH,
            "providers": PROVIDERS_ONNXRUNTIME
        },
        "inference": {
            "conf": YOLO_CONF_THRESHOLD_DET_CORNERCARD,
            "iou": 0.3,
            "imgsz": (640, 640),
            "max_det": 10,
            "classes": CLASS_NAMES,
            "project": 'analysis_error',
            "name": 'crop',
            "save": False
        }
    }

    # YOLO text region detection models
    yolo_textregiondet_configs: Dict[str, Dict[str, Any]] = {
        # Citizen card (cc) text detection model
        "cc": {
            "initial": {
                "path_or_bytes": YOLO_DET_TEXTCC_MODEL_PATH,
                "providers": PROVIDERS_ONNXRUNTIME
            },
            "inference": {
                "conf": YOLO_CONF_THRESHOLD_DET_TEXT,
                "iou": 0.3,
                "imgsz": (640, 640),
                "max_det": 10,
                "classes": FIELD_CLASS['cc'],
                "project": 'analysis_error',
                "name": 'cc_det',
                "save": False,
            }
        },
        # Citizen ID card (cccd) text detection model
        "cccd": {
            "initial": {
                "path_or_bytes": YOLO_DET_TEXTCCCD_MODEL_PATH,
                "providers": PROVIDERS_ONNXRUNTIME
            },
            "inference": {
                "conf": YOLO_CONF_THRESHOLD_DET_TEXT,
                "iou": 0.3,
                "imgsz": (640, 640),
                "max_det": 10,
                "classes": FIELD_CLASS['cccd'],
                "project": 'analysis_error',
                "name": 'cccd_det',
                "save": False,
            }
        }
    }
    
    # VietOCR configuration
    vietocr_configs: Dict[str, Union[str, Dict[str, Any]]] = {
        'cnn_onnx_path': f'{VIETOCR_REC_TEXTREGION_MODEL_PATH}/my_cnn.onnx',
        'encoder_onnx_path': f'{VIETOCR_REC_TEXTREGION_MODEL_PATH}/my_encoder.onnx',
        'decoder_onnx_path': f'{VIETOCR_REC_TEXTREGION_MODEL_PATH}/my_decoder.onnx',
        'onnx_execution_options': {
            'enable_cuda': False # 'CUDAExecutionProvider' in ort.get_available_providers()
        }
    }



# Create settings instance
settings = Settings()

if __name__ == "__main__":
    print(settings)