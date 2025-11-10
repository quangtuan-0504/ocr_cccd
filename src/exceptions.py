"""
Custom exceptions for the OCR identification card application.
"""

class OCRBaseException(Exception):
    """Base exception class for OCR application."""
    pass


class BadCaptureError(OCRBaseException):
    """Exception raised when the input image has quality issues."""
    pass


class CardDetectionError(OCRBaseException):
    """Exception raised when there's an error detecting the card in the image."""
    pass


class BadBoxError(OCRBaseException):
    """Exception raised when the detected bounding box is invalid."""
    pass

class TextRegionDetectError(OCRBaseException):
    """Exception raised when the detected bounding box text region is error."""
    pass

class TextExtractError(OCRBaseException):
    """Exception raised when the extracted texts from text regions is error."""
    pass

class ModelNotLoadedError(OCRBaseException):
    """Exception raised when a required model couldn't be loaded."""
    pass