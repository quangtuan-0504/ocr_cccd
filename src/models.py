"""
Model loading functions for the OCR identification card application.
"""

from paddleocr import DocImgOrientationClassification, TextImageUnwarping
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from configs import settings
import yaml
import os
from typing import Dict, Any, Optional

from exceptions import ModelNotLoadedError
from logger import logger


def load_yolo_cardobb_model(configs: Optional[Dict[str, Any]] = None) -> YOLO:
    """
    Load YOLOv11 model for oriented bounding box detection.

    Args:
        configs: Configuration parameters for the YOLO model.

    Returns:
        Loaded YOLOv11 model instance.

    Raises:
        ModelNotLoadedError: If the YOLOv11 model fails to load.
    """
    try:
        configs = configs or {}
        # logger.info(f"Loading YOLOv11 OBB model with config: {configs}")
        model = YOLO(**configs)
        logger.info("YOLOv11 OBB model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv11 OBB model: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load YOLOv11 OBB model: {str(e)}")


def load_paddle_imgorient_model(configs: Optional[Dict[str, Any]] = None) -> DocImgOrientationClassification:
    """
    Load PaddleOCR model for document image orientation classification.

    Args:
        configs: Configuration parameters for the PaddleOCR model.

    Returns:
        Loaded PaddleOCR orientation model instance.

    Raises:
        ModelNotLoadedError: If the PaddleOCR orientation model fails to load.
    """
    try:
        configs = configs or {}
        # logger.info(f"Loading PaddleOCR orientation model with config: {configs}")
        model = DocImgOrientationClassification(**configs)
        logger.info("PaddleOCR orientation model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load PaddleOCR orientation model: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load PaddleOCR orientation model: {str(e)}")


def load_paddle_imgunwarp_model(configs: Optional[Dict[str, Any]] = None) -> TextImageUnwarping:
    """
    Load PaddleOCR model for text image unwarping.

    Args:
        configs: Configuration parameters for the PaddleOCR model.

    Returns:
        Loaded PaddleOCR unwarping model instance.

    Raises:
        ModelNotLoadedError: If the PaddleOCR unwarping model fails to load.
    """
    try:
        configs = configs or {}
        # logger.info(f"Loading PaddleOCR unwarping model with config: {configs}")
        model = TextImageUnwarping(**configs)
        logger.info("PaddleOCR unwarping model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load PaddleOCR unwarping model: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load PaddleOCR unwarping model: {str(e)}")


def load_yolo_textdetcls_model(configs: Optional[Dict[str, Any]] = None) -> YOLO:
    """
    Load YOLOv11 model for text detection and classification.

    Args:
        configs: Configuration parameters for the YOLO model.

    Returns:
        Loaded YOLOv11 text detection and classification model instance.

    Raises:
        ModelNotLoadedError: If the YOLOv11 text detection model fails to load.
    """
    try:
        configs = configs or {}
        # logger.info(f"Loading YOLOv11 text detection model with config: {configs}")
        model = YOLO(**configs)
        logger.info("YOLOv11 text detection model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv11 text detection model: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load YOLOv11 text detection model: {str(e)}")


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