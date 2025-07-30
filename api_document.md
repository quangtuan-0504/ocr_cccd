### API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### 1. Root Endpoint

```
GET /
```

Returns a welcome message to confirm the server is running.

#### 2. OCR Processing Endpoint

```
POST /ocr-id-card/upload-image-flexible
```

This endpoint accepts either a file upload or a base64-encoded image string via form data.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | No | Image file to process (multipart/form-data) |
| image_base64 | String | No | Base64-encoded image string (form field) |
| card_type | String | No | Type of card: "cccd" (default) or "cc" |
| card_side | String | No | Side of card: "front" (default) or "back" |
| file_name | String | No | Name of file when using base64 (default: "image.jpg") |

**Note:** Either `file` or `image_base64` must be provided.

**Request Format:**
- Content-Type: `multipart/form-data` for file uploads
- Content-Type: `application/x-www-form-urlencoded` for base64 strings

**Response:**

```json
{
  "status": "success",
  "card_info": {
    "class": "cccd_front",
    "confidence": 0.98
  },
  "data": {
    "Số / No.":{"text":"001202012345","score": 0.9},
    "Họ và tên / Full name":{"text":"NGUYỄN VĂN A","score": 0.9},
    "Ngày sinh / Date of birth":{"text":"01/01/1990","score": 0.9},
    "Giới tính / Sex":{"text":"Nam","score": 0.9},
    "Quốc tịch / Nationality":{"text":"Việt Nam","score": 0.9},
    "Quê quán / Place of origin":{"text":"Hà Nội","score": 0.9},
    "Nơi thường trú / Place of residence":{"text":"123 Đường ABC, Phường XYZ, Quận 1, TP.HCM","score": 0.9},
    "Có giá trị đến / Date of expiry":{"text":"01/01/2042","score": 0.9}
  }
}
```

### Examples

#### Example 1: File Upload

```python
import requests

url = "http://localhost:8000/ocr-id-card/upload-image-flexible"
files = {"file": open("path/to/image.jpg", "rb")}
data = {"card_type": "cccd", "card_side": "front"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Example 2: Base64 Image

```python
import requests
import base64

# Read image and convert to base64
with open("path/to/image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

url = "http://localhost:8000/ocr-id-card/upload-image-flexible"
data = {
    "image_base64": encoded_string,
    "card_type": "cccd",
    "card_side": "front",
    "file_name": "my_image.jpg"
}

response = requests.post(url, data=data)
print(response.json())
```

#### Example 3: cURL File Upload

```bash
curl -X POST "http://localhost:8000/ocr-id-card/upload-image-flexible" \
     -F "file=@path/to/image.jpg" \
     -F "card_type=cccd" \
     -F "card_side=front"
```

#### Example 4: cURL Base64

```bash
curl -X POST "http://localhost:8000/ocr-id-card/upload-image-flexible" \
     -F "image_base64=data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD..." \
     -F "card_type=cccd" \
     -F "card_side=front" \
     -F "file_name=test.jpg"
```

## Error Handling

The API returns appropriate error messages for various scenarios:

- Image quality issues (blurry, dark, low contrast)
- No card detected in the image
- Wrong card type or side detected
- Processing failures

Error response example:

```json
{
  "status": "error",
  "message": "Image is too blurry (blur score: 85.23)"
}
```

## Performance Optimization

The application includes several optimizations:

- Parallel processing of text extraction
- LRU caching for expensive operations
- Thread pool for concurrent processing
- Optimized inference settings
- GPU memory management with automatic cleanup
