from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import torch
import sys
import os
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import logger
from configs import settings
from models import (
    load_yolo_cardobb_model,
    load_paddle_imgorient_model,
    load_paddle_imgunwarp_model,
    load_paddle_rec_model,
    load_yolo_textdetcls_model,
    load_vietocr_rec_onnx_model
)
from services import IdCardOCRService


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
        # Load card detection model
        logger.info('Loading OBB card detection model')
        yolo_cardobb_model = load_yolo_cardobb_model(settings.yolo_cardobb_configs.get('initial', {}))
        
        # Load image orientation model
        logger.info('Loading Paddle image orientation model')
        paddle_imgorient_model = load_paddle_imgorient_model(settings.paddle_imgorient_configs.get('initial', {}))
        logger.setLevel(logging.INFO)
        # Load image unwarping model
        logger.info('Loading Paddle unwarping image model')
        paddle_imgunwarp_model = load_paddle_imgunwarp_model(settings.paddle_imgunwarp_configs.get('initial', {}))
        # Load image recognition model
        logger.info('Loading Paddle recognition image model')
        paddle_imgrec_model = load_paddle_rec_model(settings.paddle_imgrec_configs.get('initial', {}))

        # Load text detection models for each card type
        logger.info('Loading YOLO text region detection models')
        yolo_textdetcls_model = {}
        for card_type, cfg in settings.yolo_textdetcls_configs.items():
            logger.info(f'Loading text detection model for {card_type}')
            yolo_textdetcls_model[card_type] = load_yolo_textdetcls_model(cfg.get('initial', {}))
        
        # Load text recognition model
        logger.info('Loading VietOCR recognition model')
        vietocr_rec = load_vietocr_rec_onnx_model()

        # Initialize OCR service with all models
        resources['ocr_service'] = IdCardOCRService(
            cardobb_model=yolo_cardobb_model,
            imgorient_model=paddle_imgorient_model,
            imgunwarp_model=paddle_imgunwarp_model,
            mrzrec_model=paddle_imgrec_model,
            textdetcls_model=yolo_textdetcls_model,
            vietocr_onnx=vietocr_rec
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
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            
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