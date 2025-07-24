from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from logger import logger
from configs import settings
from models import (
    load_yolo_cardobb_model,
    load_paddle_imgorient_model,
    load_paddle_imgunwarp_model,
    load_yolo_textdetcls_model,
    load_vietocr_rec_model
)
from services import IdCardOCRService


def get_services():
    """
    Load services and models for the application.
    """
    resources = {}
    try:
        logger.info(f'load obb card model')
        yolo_cardobb_model = load_yolo_cardobb_model(settings.yolo_cardobb_configs.get('initial', {}))
        logger.info(f'load paddle image orientation model')
        paddle_imgorient_model = load_paddle_imgorient_model(settings.paddle_imgorient_configs.get('initial', {}))
        logger.setLevel(logging.INFO)
        logger.info(f'load paddle unwarping image model')
        paddle_imgunwarp_model = load_paddle_imgunwarp_model(settings.paddle_imgunwarp_configs.get('initial', {}))        
        logger.info(f'load yolo text region detection model')
        yolo_textdetcls_model = {}
        for card_type, cfg in settings.yolo_textdetcls_configs.items():
            yolo_textdetcls_model[card_type] = load_yolo_textdetcls_model(cfg.get('initial', {}))
        logger.info(f'load recognition model')
        vietocr_rec_model = load_vietocr_rec_model()

        resources['ocr_service'] = IdCardOCRService(
            cardobb_model = yolo_cardobb_model,
            imgorient_model = paddle_imgorient_model,
            imgunwarp_model = paddle_imgunwarp_model,
            textdetcls_model = yolo_textdetcls_model,
            rec_model = vietocr_rec_model
        )
        logger.info("Services loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load services: {e}")
        raise
    return resources

resources: dict[str,object] = {}

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI application."""
    try:
        # Log GPU availability
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        # Load resources here
        logger.info("Loading resources...")
        service = get_services()
        resources.update(service)
        logger.info("Resources loaded successfully.")
        yield
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
    finally:
        # Clean up the ML models and release the resources
        logger.info("Cleaning up resources")
        resources.clear()
        logger.info("Resources cleared")