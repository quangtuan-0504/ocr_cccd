import os
import requests
import base64
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust to your server URL
OCR_ENDPOINT = f"{API_BASE_URL}/ocr-id-card/upload-image-flexible"
TEST_IMAGES_FOLDER = "data_test/images"  # Adjust path as needed

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string  # Return raw base64 without prefix

def test_file_upload(image_path, card_type="cccd", card_side="front"):
    """Test OCR with file upload"""
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
            data = {
                "card_type": card_type,
                "card_side": card_side
            }
            
            start_time = time.time()
            response = requests.post(OCR_ENDPOINT, files=files, data=data)
            end_time = time.time()
            
            return {
                "method": "file_upload",
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "response": response.json() if response.status_code == 200 else response.text
            }
    except Exception as e:
        return {
            "method": "file_upload",
            "error": str(e)
        }

def test_base64_upload(image_path, card_type="cccd", card_side="front"):
    """Test OCR with base64 string"""
    try:
        base64_image = encode_image_to_base64(image_path)
        
        # Use form data instead of JSON
        data = {
            "image_base64": base64_image,
            "card_type": card_type,
            "card_side": card_side,
            "file_name": os.path.basename(image_path)
        }
        
        start_time = time.time()
        response = requests.post(OCR_ENDPOINT, data=data)
        end_time = time.time()
        
        print(f"    Debug - Request data keys: {list(data.keys())}")
        print(f"    Debug - Base64 length: {len(base64_image)}")
        print(f"    Debug - Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"    Debug - Response text: {response.text[:200]}...")
        
        return {
            "method": "base64_upload",
            "status_code": response.status_code,
            "response_time": end_time - start_time,
            "response": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "method": "base64_upload",
            "error": str(e)
        }

def run_tests():
    """Run OCR tests on all images in the test folder"""
    if not os.path.exists(TEST_IMAGES_FOLDER):
        print(f"Test images folder not found: {TEST_IMAGES_FOLDER}")
        print("Please create the folder and add test images")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(TEST_IMAGES_FOLDER)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {TEST_IMAGES_FOLDER}")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Testing OCR endpoint: {OCR_ENDPOINT}")
    print("-" * 50)
    
    results = []
    
    for image_file in image_files[:75]:
        image_path = os.path.join(TEST_IMAGES_FOLDER, image_file)
        print(f"\nTesting image: {image_file}")
        
        # Test both methods
        for card_type in ["cccd"]:#["cccd", "cc"]:
            for card_side in ["front"]:#["front", "back"]:
                print(f"  Testing {card_type} - {card_side}")
                
                # Test file upload
                file_result = test_file_upload(image_path, card_type, card_side)
                file_result["image"] = image_file
                file_result["card_type"] = card_type
                file_result["card_side"] = card_side
                results.append(file_result)
                
                print(f"    File upload: {file_result.get('status_code', 'ERROR')} "
                      f"({file_result.get('response_time', 0):.2f}s)")
                
                # Test base64 upload
                base64_result = test_base64_upload(image_path, card_type, card_side)
                base64_result["image"] = image_file
                base64_result["card_type"] = card_type
                base64_result["card_side"] = card_side
                results.append(base64_result)
                
                print(f"    Base64 upload: {base64_result.get('status_code', 'ERROR')} "
                      f"({base64_result.get('response_time', 0):.2f}s)")
                
                # Small delay between requests
                time.sleep(0.5)
    
    # Save results to file
    results_file = "/workspace/ocr_cccd/test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*50)
    print(f"Testing completed. Results saved to: {results_file}")
    
    # Print summary
    successful_requests = [r for r in results if r.get('status_code') == 200]
    failed_requests = [r for r in results if r.get('status_code') != 200 or 'error' in r]
    
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful_requests)}")
    print(f"Failed: {len(failed_requests)}")
    
    if successful_requests:
        avg_time = sum(r.get('response_time', 0) for r in successful_requests) / len(successful_requests)
        print(f"Average response time: {avg_time:.2f}s")

def test_single_image(image_path, card_type="cccd", card_side="front"):
    """Test a single image with both methods"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    print(f"Testing single image: {image_path}")
    print(f"Card type: {card_type}, Card side: {card_side}")
    print("-" * 30)
    
    # Test file upload
    print("Testing file upload...")
    file_result = test_file_upload(image_path, card_type, card_side)
    print(f"Status: {file_result.get('status_code', 'ERROR')}")
    print(f"Time: {file_result.get('response_time', 0):.2f}s")
    if 'response' in file_result:
        print(f"Response: {json.dumps(file_result['response'], indent=2, ensure_ascii=False)}")
    if 'error' in file_result:
        print(f"Error: {file_result['error']}")
    
    print("\nTesting base64 upload...")
    base64_result = test_base64_upload(image_path, card_type, card_side)
    print(f"Status: {base64_result.get('status_code', 'ERROR')}")
    print(f"Time: {base64_result.get('response_time', 0):.2f}s")
    if 'response' in base64_result:
        print(f"Response: {json.dumps(base64_result['response'], indent=2, ensure_ascii=False)}")
    if 'error' in base64_result:
        print(f"Error: {base64_result['error']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test single image
        image_path = sys.argv[1]
        card_type = sys.argv[2] if len(sys.argv) > 2 else "cccd"
        card_side = sys.argv[3] if len(sys.argv) > 3 else "front"
        test_single_image(image_path, card_type, card_side)
    else:
        # Test all images in folder
        run_tests()
