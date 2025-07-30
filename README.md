# OCR Identification Card Application

A FastAPI application for extracting information from Vietnamese identification cards using computer vision and OCR techniques.

## Features

- Automatic card detection and cropping
- Image quality validation
- Card type classification (CCCD/CC)
- Card side detection (front/back)
- Text field detection and extraction
- Support for both file upload and base64 image input

## Architecture

The application uses a pipeline of state-of-the-art models:

1. **YOLOv11 OBB** - For card detection and orientation
2. **PaddleOCR** - For image orientation classification and unwarping
3. **YOLOv11** - For text field detection and classification
4. **VietOCR** - For Vietnamese text recognition

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for performance)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ocr_cccd.git
   cd ocr_cccd
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download model files (not included in repository):
   - Place YOLOv11 OBB model in `model/det_card/`
   - Place PaddleOCR models in `model/PP-LCNet_x1_0_doc_ori/` and `model/UVDoc/`
   - Place text detection models in `model/textdet/cc/` and `model/textdet/cccd/`
   - Place VietOCR model in `model/vietocr/`

## Usage

### Running the Server

Start the FastAPI server:

```bash
python src/main.py
```

The server will be available at `http://localhost:8000`.

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
    "id": "001202012345",
    "name": "NGUYỄN VĂN A",
    "date_of_birth": "01/01/1990",
    "gender": "Nam",
    "nation": "Việt Nam",
    "hometown": "Hà Nội",
    "permanent_residence": "123 Đường ABC, Phường XYZ, Quận 1, TP.HCM"
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

## License

[MIT License](LICENSE)

## Acknowledgements

- [YOLOv11](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [FastAPI](https://fastapi.tiangolo.com/)