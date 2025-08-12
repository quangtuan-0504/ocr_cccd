"""
Model loading functions for the OCR identification card application.
"""

from paddleocr import TextRecognition
import onnxruntime
import yaml
import os
from typing import Dict, Any, Optional, Tuple
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from vietocrOnnxModule.tool.config import Cfg
from vietocrOnnxModule.model.vocab import Vocab
from src.configs import settings
from src.exceptions import ModelNotLoadedError
from src.logger import logger
import torch
# import pynvml


def load_yoloonnx_4cornercarddet_model(configs: Optional[Dict[str, Any]] = None) -> onnxruntime.InferenceSession:
    """
    Load YOLOv11 onnx model for detection of 4 corner cards.

    Args:
        configs: Configuration parameters for the YOLO model.

    Returns:
        Loaded YOLOv11 model instance.

    Raises:
        ModelNotLoadedError: If the YOLOv11 model fails to load.
    """
    try:
        configs = configs or {}
        # logger.info(f"Loading YOLOv11 detection 4 corner card model with config: {configs}")
        model = onnxruntime.InferenceSession(**configs)
        logger.info("YOLOv11 detection 4 corner card model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv11 detection 4 corner card model: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load YOLOv11 detection 4 corner card model: {str(e)}")

def load_yoloonnx_textregiondet_model(configs: Optional[Dict[str, Any]] = None) -> onnxruntime.InferenceSession:
    """
    Load YOLOv11 onnx model for text region detection.

    Args:
        configs: Configuration parameters for the YOLO model.

    Returns:
        Loaded YOLOv11 text region detection model instance.

    Raises:
        ModelNotLoadedError: If the YOLOv11 text region detection model fails to load.
    """
    try:
        configs = configs or {}
        # logger.info(f"Loading YOLOv11 text detection model with config: {configs}")
        model = onnxruntime.InferenceSession(**configs)
        logger.info("YOLOv11 text detection model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv11 text detection model: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load YOLOv11 text detection model: {str(e)}")
    
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
    
def load_vietocronnx_rec_model() -> Tuple:
    """
    Load VietOCR model for text recognition.

    Returns:
        Predictor: Loaded onnx VietOCR model.

    Raises:
        RuntimeError: If the VietOCR model fails to load.
    """
    try:
        # # Check cuda availability
        # try:
        #     pynvml.nvmlInit()
        #     cuda_check = pynvml.nvmlDeviceGetCount() > 0
        #     pynvml.nvmlShutdown()
        # except Exception:
        #     cuda_check = False
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