from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import base64
import io
from PIL import Image
import os
from typing import Optional
from load_resources import resources


router = APIRouter(
    prefix="/ocr-id-card",
    tags=["OCR"]
)
@router.post("/upload-image-flexible", response_model=dict)
async def ocr_id_card(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = None,
    card_type: str|None = None,
    card_side: str|None = None,
    file_name: str|None = "image.jpg"
):
    """
    Endpoint flexible - can be received both file or base64 string
    """
    try:

        image = None
        
        if file:
            # process file upload
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            file_name = file.filename
            
        elif image_base64:
            # process base64
            if "," in image_base64:
                base64_data = image_base64.split(",")[1]
            else:
                base64_data = image_base64
            
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
        else:
            raise HTTPException(status_code=400, detail="Vui lòng cung cấp file hoặc base64 string")
        
        # predict using the OCR service
        result = resources['ocr_service'].process_id_card(
            image = image,
            required_type = card_type,
            required_side = card_side,
        )       
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý ảnh: {str(e)}")
