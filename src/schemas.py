from pydantic import BaseModel, Field
from typing import List, Optional

class OCRResponse(BaseModel):
    """Response model for OCR results."""
    texts: dict[str, str]