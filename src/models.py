"""
Model loading functions for the OCR identification card application.
"""

from paddleocr import DocImgOrientationClassification, TextImageUnwarping, TextRecognition
from ultralytics import YOLO
from configs import settings
import onnxruntime
import yaml
import os
from typing import Dict, Any, Optional, Tuple
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from vietocrOnnxModule.tool.config import Cfg
from vietocrOnnxModule.model.vocab import Vocab
from exceptions import ModelNotLoadedError
from logger import logger
import torch


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

def load_paddle_rec_model(configs: Optional[Dict[str, Any]] = None) -> TextRecognition:
    """
    Load PaddleOCR model for text image recognition.

    Args:
        configs: Configuration parameters for the PaddleOCR model.

    Returns:
        Loaded PaddleOCR recognition model instance.

    Raises:
        ModelNotLoadedError: If the PaddleOCR recognition model fails to load.
    """
    try:
        configs = configs or {}
        # logger.info(f"Loading PaddleOCR recognition model with config: {configs}")
        model = TextRecognition(**configs)
        logger.info("PaddleOCR recognition model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load PaddleOCR recognition model: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load PaddleOCR recognition model: {str(e)}")

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

    
def load_vietocr_rec_onnx_model() -> Tuple:
    """
    Load VietOCR model for text recognition.

    Returns:
        Predictor: Loaded onnx VietOCR model.

    Raises:
        RuntimeError: If the VietOCR model fails to load.
    """
    try:
        # Check cuda availability
        cuda_check = torch.cuda.is_available()
        # Enable cuda if available
        enable_cuda = settings.vietocr_configs.get('onnx_execution_options', {}).get('enable_cuda', False) and cuda_check
        # Define providers based on GPU availability
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if enable_cuda else ['CPUExecutionProvider']
        logger.info(f"Using ONNX Runtime providers for VietOCR: {providers}")

        # Enable CUDA optimization
        if enable_cuda:
            # Get ONNX execution options from settings
            sess_options = onnxruntime.SessionOptions()
            
            # Enable CUDA optimization
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set execution mode
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        else:
            sess_options = None
        
        # Create inference sessions with GPU support
        cnn_session = onnxruntime.InferenceSession(
            settings.vietocr_configs['cnn_onnx_path'],
            sess_options=sess_options,
            providers=providers
        )
        encoder_session = onnxruntime.InferenceSession(
            settings.vietocr_configs['encoder_onnx_path'],
            sess_options=sess_options,
            providers=providers
        )
        decoder_session = onnxruntime.InferenceSession(
            settings.vietocr_configs['decoder_onnx_path'],
            sess_options=sess_options,
            providers=providers
        )
        
        session = (cnn_session, encoder_session, decoder_session)
        config = Cfg.load_config_from_file(
            fname = './src/vietocrOnnxModule/config/vgg-seq2seq.yml',
            base_file = './src/vietocrOnnxModule/config/base.yml'
        )
        # Set device in config
        config['device'] = 'cuda' if enable_cuda else 'cpu'
        vocab = Vocab(config['vocab'])
        return session, vocab, config
    except Exception as e:
        logger.error(f"Failed to load VietOCR text recognition model: {str(e)}")
        raise RuntimeError(f"Failed to load VietOCR text recognition model: {str(e)}")