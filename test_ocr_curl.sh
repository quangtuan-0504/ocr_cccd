#!/bin/bash

# Cấu hình
API_URL="http://localhost:8000/ocr"
IMAGE_PATH="path/to/your/id_card.jpg"

# Encode ảnh sang base64
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

# Gửi request
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$IMAGE_BASE64\",
    \"request_id\": \"test_curl_001\"
  }" \
  | jq '.'
