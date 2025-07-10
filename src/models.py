from paddleocr import PaddleOCR
from ultralytics import YOLO
from configs import settings


def load_paddleocr(configs: dict = None) -> PaddleOCR:
    """
    Load PaddleOCR model with Vietnamese language support.
    """
    try:
        ocr = PaddleOCR(
            **configs
        )
        return ocr
    except Exception as e:
        raise RuntimeError(f"Failed to load PaddleOCR model: {e}")
def load_yoloobb(configs: dict = None)-> YOLO:
    """
    Load YOLOv11 model for oriented bouding box.
    """
    try:
        model = YOLO(
            **configs
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv8 model: {e}")