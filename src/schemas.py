"""
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class OCRResponse(BaseModel):
    """Response model for OCR API endpoints."""
    status: str = Field(..., description="Status of the OCR process (success or error)")
    message: Optional[str] = Field(None, description="Error message if status is error")
    card_info: Optional[Dict[str, Any]] = Field(None, description="Information about the detected card")
    data: Optional[Dict[str, Any]] = Field(None, description="Extracted text data from the card")