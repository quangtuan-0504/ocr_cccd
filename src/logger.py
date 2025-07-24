"""
Logger configuration for the OCR identification card application.
"""

import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create and configure logger
logger = logging.getLogger("ocr_app")
logger.setLevel(logging.INFO)