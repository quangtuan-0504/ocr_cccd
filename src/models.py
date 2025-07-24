from paddleocr import DocImgOrientationClassification, TextImageUnwarping
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from configs import settings
import yaml
from typing import Optional

def load_yolo_cardobb_model(configs: Optional[dict] = None) -> YOLO:
    """
    Load YOLOv11 model for oriented bounding box detection.

    Args:
        configs (dict, optional): Configuration parameters for the YOLO model.

    Returns:
        YOLO: Loaded YOLOv11 model instance.

    Raises:
        RuntimeError: If the YOLOv11 model fails to load.
    """
    try:
        configs = configs or {}
        return YOLO(**configs)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv11 oriented bounding box model: {str(e)}")

def load_paddle_imgorient_model(configs: Optional[dict] = None) -> DocImgOrientationClassification:
    """
    Load PaddleOCR model for document image orientation classification.

    Args:
        configs (dict, optional): Configuration parameters for the PaddleOCR model.

    Returns:
        DocImgOrientationClassification: Loaded PaddleOCR orientation model instance.

    Raises:
        RuntimeError: If the PaddleOCR orientation model fails to load.
    """
    try:
        configs = configs or {}
        return DocImgOrientationClassification(**configs)
    except Exception as e:
        raise RuntimeError(f"Failed to load PaddleOCR orientation model: {str(e)}")

def load_paddle_imgunwarp_model(configs: Optional[dict] = None) -> TextImageUnwarping:
    """
    Load PaddleOCR model for text image unwarping.

    Args:
        configs (dict, optional): Configuration parameters for the PaddleOCR model.

    Returns:
        TextImageUnwarping: Loaded PaddleOCR unwarping model instance.

    Raises:
        RuntimeError: If the PaddleOCR unwarping model fails to load.
    """
    try:
        configs = configs or {}
        return TextImageUnwarping(**configs)
    except Exception as e:
        raise RuntimeError(f"Failed to load PaddleOCR unwarping model: {str(e)}")

def load_yolo_textdetcls_model(configs: Optional[dict] = None) -> YOLO:
    """
    Load YOLOv11 model for text detection and classification.

    Args:
        configs (dict, optional): Configuration parameters for the YOLO model.

    Returns:
        YOLO: Loaded YOLOv11 text detection and classification model instance.

    Raises:
        RuntimeError: If the YOLOv11 text detection model fails to load.
    """
    try:
        configs = configs or {}
        return YOLO(**configs)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv11 text detection model: {str(e)}")

def load_vietocr_rec_model() -> Predictor:
    """
    Load VietOCR model for text recognition.

    Returns:
        Predictor: Loaded VietOCR model instance.

    Raises:
        RuntimeError: If the VietOCR model fails to load.
    """
    try:
        with open(settings.vietocr_configs['base_cf'], encoding="utf-8") as f:
            base_cf = yaml.safe_load(f)
        with open(settings.vietocr_configs['cf'], encoding="utf-8") as f:
            cus_cf = yaml.safe_load(f)
        
        base_cf.update(cus_cf)
        config = Cfg(base_cf)
        config['weights'] = settings.vietocr_configs['model_path']
        vietocr_model = Predictor(config)
        return vietocr_model
    except Exception as e:
        raise RuntimeError(f"Failed to load VietOCR text recognition model: {str(e)}")