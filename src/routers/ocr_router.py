from fastapi import APIRouter, HTTPException
from typing import Optional
import base64
import io
from PIL import Image
import time
from logger import logger
from src.load_resources import resources
from schemas import BaseResponse, ResponseMessage
from exceptions import CardDetectionError, BadCaptureError, TextRegionDetectError, TextExtractError , BadBoxError
from pydantic import BaseModel
from dataclasses import asdict

class OCRRequest(BaseModel):
    image_base64: str
    request_id: Optional[str] = None


router = APIRouter(
    tags=["OCR"]
)

@router.post("/ocr")
async def ocr_id_card(request: OCRRequest):
    """
    Process an ID card image and extract information.

    Parameters: 
    - request: JSON body containing:
        - image_base64: Base64 encoded image string
        - request_id: Optional request identifier

    Returns:
    - OCRResponse with status, card information and extracted data
    """
    try:
        image = None

        try:
            # Remove data URL prefix if present
            if "," in request.image_base64:
                base64_data = request.image_base64.split(",")[1]
            else:
                base64_data = request.image_base64

            # Decode and open image
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
        except ValueError as ve:
            return asdict(
                BaseResponse(
                    data=[],
                    errorCode=400,
                    errorMessage=ResponseMessage.input_error(str(ve))
                )
            )

        # Predict using the OCR service
        start = time.time()
        result = resources['ocr_service'].process_id_card(image=image)
        processing_time = time.time() - start

        logger.info(f'Request ID: {request.request_id}, Time: {processing_time:.3f}s')

        return asdict(BaseResponse(
                data=result,
                errorCode=200, 
                errorMessage=ResponseMessage.SUCCESS
            )
        )
    except BadCaptureError as e:
        logger.error(f"Bad Capture issue: {str(e)}")
        return asdict(BaseResponse(
                data=[],
                errorCode=400, 
                errorMessage=ResponseMessage.bad_capture(str(ve))
            )
        )
    except CardDetectionError as e:
        logger.error(f"Card Detect issue: {str(e)}")
        return asdict(BaseResponse(
                data=[],
                errorCode=400, 
                errorMessage=ResponseMessage.card_detect_error(str(ve))
            )
        )
    except BadBoxError as e:
        logger.error(f"Bad Box issue: {str(e)}")
        return asdict(BaseResponse(
                data=[],
                errorCode=400, 
                errorMessage=ResponseMessage.bad_box(str(ve))
            )
        )
    except TextRegionDetectError as e:
        logger.error(f"Text Region Detect issue: {str(e)}")
        return asdict(BaseResponse(
                data=[],
                errorCode=400, 
                errorMessage=ResponseMessage.text_region_det_error(str(ve))
            )
        )
    except TextExtractError as e:
        logger.error(f"Text Extract issue: {str(e)}")
        return asdict(BaseResponse(
                data=[],
                errorCode=400, 
                errorMessage=ResponseMessage.text_extract_error(str(ve))
            )
        )
    except ValueError as ve:
        return asdict(BaseResponse(
                data=[],
                errorCode=400, 
                errorMessage=ResponseMessage.process_img_error(str(ve))
            )
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return asdict(
            BaseResponse(
                data=[],
                errorCode=500,
                errorMessage=ResponseMessage.error_occurred(e)
            )
        )
    except HTTPException:
        raise