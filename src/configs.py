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
        # 0: 'top_left',
        # 1: 'top_right',
        # 2: 'bottom_right',
        # 3: 'bottom_left',
        # 4: 'cc_front',
        # 5: 'cc_back',
        # 6: "cccd_front",
        # 7: "cccd_back",
        # 8: "other_card"

        # toy model class names
        0: 'bottom_left',
        1: 'bottom_right',
        2: 'cc_back',
        3: 'cc_front',
        4: 'top_left',
        5: 'top_right',
        6: "cccd_front",
        7: "cccd_back",
        8: "other_card"
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
    
    YOLO_DET_4CORNERCARD_MODEL_PATH: str = "model/det_card/best.pt"
    YOLO_DET_TEXTCC_MODEL_PATH: str = "model/detcls_text_region/best.pt"
    YOLO_DET_TEXTCCCD_MODEL_PATH: str = "model/detcls_text_region/best.pt"

    YOLO_CONF_THRESHOLD_DET_CORNERCARD: float = 0.35
    YOLO_CONF_THRESHOLD_DET_TEXT: float = 0.5

    PADDLE_REC_TEXTREGION_MODEL_PATH: str = "model/en_PP-OCRv4_mobile_rec"
    VIETOCR_REC_TEXTREGION_MODEL_PATH: str = "model/vietocr/"
    # YOLOv11 detection 4 corner card model for card detection
    yolo_4cornercarddet_configs: Dict[str, Dict[str, Any]] = {
        "initial": {
            "model": YOLO_DET_4CORNERCARD_MODEL_PATH,
            "task": "obb"
        },
        "inference": {
            "conf": YOLO_CONF_THRESHOLD_DET_CORNERCARD,
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

    # YOLO text region detection models
    yolo_textregiondet_configs: Dict[str, Dict[str, Any]] = {
        # Citizen card (cc) text detection model
        "cc": {
            "initial": {
                "model": YOLO_DET_TEXTCC_MODEL_PATH,
                "task": "detect"
            },
            "inference": {
                "conf": YOLO_CONF_THRESHOLD_DET_TEXT,
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
                "save": False # if you want to save results image, you must pass img input as pil image
                # Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            }
        },
        # Citizen ID card (cccd) text detection model
        "cccd": {
            "initial": {
                "model": YOLO_DET_TEXTCCCD_MODEL_PATH,
                "task": "detect"
            },
            "inference": {
                "conf": YOLO_CONF_THRESHOLD_DET_TEXT,
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
                "save": False
            }
        }
    }
    # PaddleOCR image recognition model
    paddle_imgrec_configs: Dict[str, Dict[str, Any]] = {
        "initial": {
            "model_name": "en_PP-OCRv4_mobile_rec",
            "model_dir": PADDLE_REC_TEXTREGION_MODEL_PATH,
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "enable_hpi": False,
        },
        "inference": {
            "batch_size": 1
        }
    }
    # VietOCR configuration
    vietocr_configs: Dict[str, Union[str, Dict[str, Any]]] = {
        'cnn_onnx_path': f'{VIETOCR_REC_TEXTREGION_MODEL_PATH}/cnn.onnx',
        'encoder_onnx_path': f'{VIETOCR_REC_TEXTREGION_MODEL_PATH}/encoder.onnx',
        'decoder_onnx_path': f'{VIETOCR_REC_TEXTREGION_MODEL_PATH}/decoder.onnx',
        'onnx_execution_options': {
            # 'enable_cuda': torch.cuda.is_available()
            'enable_cuda': False
        }
    }


# Create settings instance
settings = Settings()

if __name__ == "__main__":
    print(settings)