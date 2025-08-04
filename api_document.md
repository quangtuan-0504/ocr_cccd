# OCR ID Card API Documentation

## Overview
This API provides OCR (Optical Character Recognition) functionality for Vietnamese ID cards (CCCD and CC). It can process images uploaded as files or base64 encoded strings and extract text information from the cards.

## Base URL
```
/ocr-id-card
```

## Endpoints

### POST /upload-image-flexible

Process an ID card image and extract information using advanced OCR technology.

#### Request Parameters

The API accepts both form-data and multipart/form-data requests with the following parameters:

| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `file` | UploadFile | No* | Image file upload (JPEG, PNG, etc.) | None |
| `image_base64` | string | No* | Base64 encoded image string | None |
| `card_type` | string | No | Type of card (cccd, cc) | Auto-detected |
| `card_side` | string | No | Side of card (front, back) | Auto-detected |
| `file_name` | string | No | Name of file (used when image_base64 is provided) | "image.jpg" |
| `mime` | string | No | MIME type of the image | None |
| `postname` | string | No | Additional metadata | None |
| `sessionID` | string | No | Session identifier for tracking | None |
| `deviceInfo` | string | No | Device information | None |

*Note: Either `file` or `image_base64` must be provided.

#### Request Examples

**File Upload:**
```bash
curl -X POST "http://localhost:8000/ocr-id-card/upload-image-flexible" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/id_card.jpg" \
  -F "sessionID=12345"
```

**Base64 Upload:**
```bash
curl -X POST "http://localhost:8000/ocr-id-card/upload-image-flexible" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "image_base64=data:image/jpeg;base64,/9j/4AAQSkZJRgABA..." \
  -d "file_name=id_card.jpg" \
  -d "sessionID=12345"
```

#### Response Format

The API returns a JSON response with the following structure:

```json
{
  "errorCode": "string",
  "errorMessage": "string", 
  "data": [
    {
      "card_type": "string",
      "card_side": "string",
      "extracted_fields": {
        "field_name": "extracted_text",
        ...
      }
    }
  ]
}
```

#### Success Response (errorCode: "0")

```json
{
  "errorCode": "0",
  "errorMessage": "no error",
  "data": [
    {    
        "id": "",
    
        "id_prob": "",
    
        "name": "",
    
        "name_prob": "",
    
        "dob": "",
    
        "dob_prob": "",
    
        "sex": "",
    
        "sex_prob": "",
    
        "nationality": "",
    
        "nationality_prob": "",
    
        "home": "",
    
        "home_prob": "",
    
        "address": "",
    
        "address_prob": "",
    
        "doe": "",
    
        "doe_prob": "",
    
        "religion": "",
    
        "religion_prob": "",
    
        "ethnicity": "",
    
        "ethnicity_prob": "",
    
        "features": "",
    
        "features_prob": "",
    
        "issue_date": "",
    
        "issue_date_prob": "",
    
        "issue_loc": "",
    
        "issue_loc_prob": "",
    
        "back_type": "N/A",
    
        "overall_score": "",
    
        "type": "",
    
        "type_new": "",
    
        "mrz": "N/A",
    
        "mrz_prob": "N/A",
    
        "mrz_details": {
    
            "id": "N/A",
    
            "name": "N/A",
    
            "doe": "N/A",
    
            "dob": "N/A",
    
            "nationality": "N/A",
    
            "sex": "N/A"
    
        },
    
        "custAddress": {
    
            "province": "",
    
            "district": "",
    
            "ward": "",
    
            "street": ""
    
        }
    }
  ]
}
```

#### Error Responses

**Error Code "1" - Image Quality/Processing Issues:**
```json
{
  "errorCode": "1", 
  "errorMessage": "Image quality is too low or processing failed",
  "data": []
}
```

**Error Code "2" - Card Detection Issues:**
```json
{
  "errorCode": "2",
  "errorMessage": "No card detected in image or invalid card type", 
  "data": []
}
```

**Error Code "3" - Text Extraction Issues:**
```json
{
  "errorCode": "3",
  "errorMessage": "Text extraction failed",
  "data": []
}
```

## Supported Card Types

- **CCCD (Căn cước công dân)**: Vietnamese National ID Card
- **CC (Chứng minh nhân dân)**: Vietnamese Identity Card

## Supported Card Sides

- **Front**: Contains personal information
- **Back**: Contains additional details and signatures

## Image Requirements

- **Format**: JPEG, PNG, or other common image formats
- **Quality**: High resolution recommended for better OCR accuracy
- **Orientation**: Any orientation (auto-corrected)
- **Size**: Maximum file size depends on server configuration

## Processing Pipeline

1. **Image Validation**: Checks image quality and format
2. **Card Detection**: Uses YOLOv11 OBB to detect and crop the card
3. **Orientation Correction**: Automatically rotates the image to correct orientation
4. **Image Unwarping**: Corrects perspective distortion
5. **Text Detection**: Identifies text regions using YOLO models
6. **Text Recognition**: Extracts text using VietOCR ONNX models
7. **Field Mapping**: Maps extracted text to appropriate card fields

## Performance

- **Average Processing Time**: 1-3 seconds per image
- **Concurrent Requests**: Supported with thread pool processing
- **Memory Optimization**: GPU memory automatically cleared after processing

## Error Handling

The API includes comprehensive error handling for:
- Invalid image formats
- Poor image quality
- Card detection failures
- Text extraction issues
- Server processing errors

## Security Considerations

- Images are processed in memory and not permanently stored
- Session IDs can be used for request tracking
- Device information can be logged for audit purposes

## Rate Limiting

Rate limiting may apply depending on server configuration. Contact your administrator for specific limits.

## Support

For technical support or issues, please check the server logs or contact the development team.
