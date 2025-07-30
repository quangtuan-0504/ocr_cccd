from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional
import base64
import io
from PIL import Image
import time
from logger import logger
from load_resources import resources
from schemas import OCRResponse


router = APIRouter(
    prefix="/ocr-id-card",
    tags=["OCR"]
)

@router.post("/upload-image-flexible")
async def ocr_id_card(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    card_side: Optional[str] = Form(None),
    file_name: Optional[str] = Form("image.jpg"),
    mime: Optional[str] = Form(None),
    postname: Optional[str] = Form(None),
    sessionID: Optional[str] = Form(None),
    deviceInfo: Optional[str] = Form(None),
):
    """
    Process an ID card image and extract information.
    
    Parameters:
    - file: Image file upload
    - image_base64: Base64 encoded image string
    - card_type: Type of card (cccd, cc)
    - card_side: Side of card (front, back)
    - file_name: Name of file (used when image_base64 is provided)
    
    Returns:
    - OCRResponse with status, card information and extracted data
    """
    try:
        image = None
        
        if file:
            # Process file upload
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            file_name = file.filename
            
        elif image_base64:
            # Process base64
            try:
                if "," in image_base64:
                    base64_data = image_base64.split(",")[1]
                else:
                    base64_data = image_base64
                
                image_data = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
            
        else:
            raise HTTPException(status_code=400, detail="Please provide either a file or base64 string")
        
    
        
        # Predict using the OCR service
        start = time.time()
        result = resources['ocr_service'].process_id_card(
            image=image,
        
        )
        processing_time = time.time() - start
        logger.info(f'Time per request: {processing_time:.3f}s')
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
