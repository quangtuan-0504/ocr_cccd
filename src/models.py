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
        logger.info(f"Loading YOLOv11 OBB model with config: {configs}")
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
        logger.info(f"Loading PaddleOCR orientation model with config: {configs}")
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
        logger.info(f"Loading PaddleOCR unwarping model with config: {configs}")
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
        logger.info(f"Loading YOLOv11 text detection model with config: {configs}")
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
        Loaded VietOCR model instance.

    Raises:
        ModelNotLoadedError: If the VietOCR model fails to load.
        FileNotFoundError: If the model files are not found.
    """
    try:
        logger.info("Loading VietOCR text recognition model")
        
        # Create config from the simplified settings
        config = Cfg.load_config_from_name('vgg_transformer')
        
        # Check if the weights file exists
        weights_path = settings.vietocr_configs.get('weights')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"VietOCR weights file not found at: {weights_path}")
            
        # Set device and weights path
        config['device'] = settings.vietocr_configs.get('device', 'cpu')
        config['weights'] = weights_path
        
        # Custom config file if provided
        config_path = settings.vietocr_configs.get('config')
        if config_path and os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                custom_config = yaml.safe_load(f)
                config.update(custom_config)
        
        vietocr_model = Predictor(config)
        logger.info("VietOCR text recognition model loaded successfully")
        return vietocr_model
        
    except FileNotFoundError as e:
        logger.error(f"VietOCR model file not found: {str(e)}")
        raise ModelNotLoadedError(f"VietOCR model file not found: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load VietOCR text recognition model: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load VietOCR text recognition model: {str(e)}")