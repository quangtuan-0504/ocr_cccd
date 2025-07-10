from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from logger import logger
from configs import settings
from models import load_paddleocr, load_yoloobb
from services import IdCardOCRService


def get_services():
    """
    Load services and models for the application.
    """
    resources = {}
    try:
        paddleocr = load_paddleocr(settings.paddleocr_configs.get('initial', {}))
        yoloobb = load_yoloobb(settings.yoloobb_configs.get('initial', {}))
        resources['ocr_service'] = IdCardOCRService(
            yolo_model=yoloobb,
            paddleocr_model=paddleocr
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