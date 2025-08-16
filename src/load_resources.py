from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import sys
import os
from typing import Dict, Any
import onnxruntime as ort
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import logger
from src.configs import settings
from src.models import (
    load_yoloonnx_4cornercarddet_model,
    load_yoloonnx_textregiondet_model,
    load_paddle_rec_model,
    load_vietocronnx_rec_model
)
from services import IdCardOCRService
from src.yoloOnnxModule.prediction import YOLOv11OnnxRuntime

def get_services() -> Dict[str, Any]:
    """
    Load all required models and services for the application.
    
    Returns:
        Dictionary containing initialized services.
        
    Raises:
        Exception: If any model fails to load.
    """
    resources = {}
    try:
        logger.info('Loading YOLO corner card detection model')
        model_4cornercarddet = load_yoloonnx_4cornercarddet_model(settings.yolo_4cornercarddet_configs.get('initial', {}))
        yolo_4cornercarddet_model = YOLOv11OnnxRuntime(model_4cornercarddet)
        logger.setLevel(logging.INFO)
    
        logger.info('Loading YOLO text region detection models')
        yolo_textregiondet_model = {}
        for card_type, cfg in settings.yolo_textregiondet_configs.items():
            logger.info(f'Loading text detection model for {card_type}')
            model_dettextregion = load_yoloonnx_textregiondet_model(cfg.get('initial', {}))
            yolo_textregiondet_model[card_type] = YOLOv11OnnxRuntime(model_dettextregion)

        logger.info('Loading paddle recognition model')
        paddle_imgrec_model = load_paddle_rec_model(settings.paddle_imgrec_configs.get('initial', {}))

        logger.info('Loading VietOCR recognition model')
        vietocr_rec_model = load_vietocronnx_rec_model()

        # Initialize OCR service with all models
        resources['ocr_service'] = IdCardOCRService(
            yolo_4cornercarddet_model=yolo_4cornercarddet_model,
            yolo_textregiondet_model=yolo_textregiondet_model,
            paddle_imgrec_model=paddle_imgrec_model,
            vietocr_rec_model=vietocr_rec_model
        )
        logger.info("All services loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load services: {e}")
        raise
    return resources


# Global resources dictionary
resources: Dict[str, Any] = {}


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application.
    
    Handles resource initialization and cleanup.
    
    Args:
        app: FastAPI application instance.
    """
    try:
        # Log GPU availability
        logger.info(f"GPU available: {'CUDAExecutionProvider' in ort.get_available_providers()}")
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            logger.info(f"GPU device: {ort.get_device()}")

        # Load resources
        logger.info("Loading application resources...")
        service = get_services()
        resources.update(service)
        logger.info("Resources loaded successfully.")
        
        yield
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        resources.clear()
        logger.info("Resources cleared successfully.")