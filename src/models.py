from paddleocr import DocImgOrientationClassification, TextImageUnwarping
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from configs import settings


def load_yolo_cardobb_model(configs: dict = None)-> YOLO:
    """
    Load YOLOv11 model for oriented bouding box.
    """
    try:
        model = YOLO(
            **configs
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv11 model: {e}")
def load_paddle_imgorient_model(configs: dict = None)-> DocImgOrientationClassification:
    """
    Load paddle image orientation model.
    """
    try:
        model = DocImgOrientationClassification(
            **configs
            # model_name="PP-LCNet_x1_0_doc_ori"
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load image orientation model: {e}")
def load_paddle_imgunwarp_model(configs: dict = None)-> TextImageUnwarping:
    """
    Load paddle unwarping image model.
    """
    try:
        model = TextImageUnwarping(
            **configs
            # model_name="UVDoc"
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load image unwarping model: {e}")
def load_yolo_textdetcls_model(configs: dict = None)-> YOLO:
    """
    Load yolov11 text detection, classify field.
    """
    try:
        model = YOLO(
            **configs
            # model_name=""
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load image unwarping model: {e}")
def load_vietocr_rec_model() -> Predictor:
    """
    Load vietoce for text recognition.
    """
    try:
        # --- Khởi tạo VietOCR ---
        config = Cfg.load_config_from_name('vgg_seq2seq')
        # config['weights'] = 'weights/transformerocr.pth'  # tải từ VietOCR repo
        config['device'] = 'cuda'  # hoặc 'cuda' nếu có GPU
        vietocr_model = Predictor(config)
        return vietocr_model
    except Exception as e:
        raise RuntimeError(f"Failed to load image recognition model: {e}")
