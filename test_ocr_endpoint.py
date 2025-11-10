import requests
import base64
import json
from pathlib import Path

# Cấu hình
API_URL = "http://localhost:8000/ocr"  # Thay đổi URL nếu cần
IMAGE_PATH = "path/to/your/id_card.jpg"  # Đường dẫn đến ảnh CCCD test

def encode_image_to_base64(image_path: str) -> str:
    """Đọc ảnh và chuyển sang base64"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def test_ocr_endpoint(image_path: str, request_id: str = "test_001"):
    """
    Gửi request test đến OCR endpoint
    
    Args:
        image_path: Đường dẫn đến ảnh cần test
        request_id: ID để tracking request
    """
    try:
        # Kiểm tra file tồn tại
        if not Path(image_path).exists():
            print(f"❌ Không tìm thấy file: {image_path}")
            return
        
        print(f"📷 Đang đọc ảnh: {image_path}")
        image_base64 = encode_image_to_base64(image_path)
        
        # Tạo payload
        payload = {
            "image_base64": image_base64,
            "request_id": request_id
        }
        
        print(f"🚀 Đang gửi request đến: {API_URL}")
        print(f"📝 Request ID: {request_id}")
        
        # Gửi POST request
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # In kết quả
        print(f"\n📊 Status Code: {response.status_code}")
        print(f"⏱️  Response Time: {response.elapsed.total_seconds():.3f}s")
        
        # Parse JSON response
        result = response.json()
        print(f"\n📄 Response Body:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Kiểm tra kết quả
        if result.get('errorCode') == 200:
            print(f"\n✅ OCR thành công!")
            if result.get('data'):
                print(f"📋 Thông tin trích xuất:")
                for key, value in result['data'].items():
                    print(f"  - {key}: {value}")
        else:
            print(f"\n⚠️  Có lỗi xảy ra: {result.get('errorMessage')}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Không thể kết nối đến server tại {API_URL}")
        print("   Đảm bảo server đang chạy!")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")

if __name__ == "__main__":
    # Test với ảnh mẫu
    test_ocr_endpoint(
        image_path=IMAGE_PATH,
        request_id="test_request_001"
    )
    
    # Có thể test nhiều ảnh
    # test_ocr_endpoint("path/to/image2.jpg", "test_request_002")
