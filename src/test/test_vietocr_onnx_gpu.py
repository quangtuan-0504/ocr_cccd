import os
import sys
import torch
import numpy as np
import onnxruntime
from PIL import Image
import cv2
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.vietocrOnnxModule.tool.translate import translate_onnx, process_input
from src.vietocrOnnxModule.model.vocab import Vocab
from src.vietocrOnnxModule.tool.config import Cfg

def test_vietocr_onnx_gpu():
    """Test VietOCR ONNX model with GPU acceleration"""
    print("Testing VietOCR ONNX with GPU acceleration")
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Define providers based on GPU availability
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    print(f"Using ONNX Runtime providers: {providers}")
    
    # Get available providers from ONNX Runtime
    available_providers = onnxruntime.get_available_providers()
    print(f"Available ONNX Runtime providers: {available_providers}")
    
    # Configure ONNX Runtime session options
    sess_options = onnxruntime.SessionOptions()
    
    if torch.cuda.is_available():
        # Set GPU memory limit
        # sess_options.gpu_mem_limit = 2 * 1024 * 1024 * 1024  # 2GB
        
        # Set arena extend strategy
        # sess_options.set_arena_extend_strategy(onnxruntime.ArenaExtendStrategy.NEXT_POWER_OF_TWO)
        
        # Enable CUDA optimization
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    
    # Load ONNX models
    try:
        print("Loading ONNX models...")
        cnn_session = onnxruntime.InferenceSession(
            'model/vietocr/cnn.onnx',
            sess_options=sess_options,
            providers=providers
        )
        encoder_session = onnxruntime.InferenceSession(
            'model/vietocr/encoder.onnx',
            sess_options=sess_options,
            providers=providers
        )
        decoder_session = onnxruntime.InferenceSession(
            'model/vietocr/decoder.onnx',
            sess_options=sess_options,
            providers=providers
        )
        
        print("Models loaded successfully!")
        
        # Load configuration and vocabulary
        config = Cfg.load_config_from_file(
            fname='./src/vietocrOnnxModule/config/vgg-seq2seq.yml',
            base_file='./src/vietocrOnnxModule/config/base.yml'
        )
        vocab = Vocab(config['vocab'])
        
        # Load a sample image
        sample_img_path = './src/vietocrOnnxModule/sample/35944.png'
        if os.path.exists(sample_img_path):
            print(f"Loading sample image: {sample_img_path}")
            image = cv2.imread(sample_img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            img = process_input(
                image, 
                config['dataset']['image_height'],
                config['dataset']['image_min_width'], 
                config['dataset']['image_max_width']
            )
            
            # Run inference
            print("Running inference...")
            start_time = time.time()
            
            session = (cnn_session, encoder_session, decoder_session)
            result = translate_onnx(img, session)[0].tolist()
            text = vocab.decode(result)
            
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.4f} seconds")
            print(f"Recognized text: {text}")
            
            # Get provider info
            for sess in [cnn_session, encoder_session, decoder_session]:
                print(f"Session provider: {sess.get_providers()}")
                
            return True
        else:
            print(f"Sample image not found: {sample_img_path}")
            return False
    
    except Exception as e:
        print(f"Error testing VietOCR ONNX with GPU: {str(e)}")
        return False

if __name__ == "__main__":
    test_vietocr_onnx_gpu()