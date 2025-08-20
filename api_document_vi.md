# Tài liệu API OCR Thẻ Căn cước

## Tổng quan
API này cung cấp chức năng OCR (Nhận dạng ký tự quang học) cho thẻ căn cước Việt Nam (CCCD và CMND). Nó có thể xử lý hình ảnh được tải lên dưới dạng file hoặc chuỗi mã hóa base64 và trích xuất thông tin văn bản từ thẻ.

## URL gốc
```
/ocr-id-card
```

## Endpoints

### POST /upload-image-flexible

Xử lý hình ảnh thẻ căn cước và trích xuất thông tin sử dụng công nghệ OCR tiên tiến.

#### Tham số yêu cầu

API chấp nhận cả yêu cầu form-data và multipart/form-data với các tham số sau:

| Tham số | Kiểu | Bắt buộc | Mô tả | Mặc định |
|---------|------|----------|-------|----------|
| `file` | UploadFile | Không* | Tải lên file hình ảnh (JPEG, PNG, v.v.) | Không có |
| `image_base64` | string | Không* | Chuỗi hình ảnh mã hóa Base64 | Không có |
| `card_type` | string | Không | Loại thẻ (cccd, cc) | Tự động phát hiện |
| `card_side` | string | Không | Mặt thẻ (front, back) | Tự động phát hiện |
| `file_name` | string | Không | Tên file (dùng khi cung cấp image_base64) | "image.jpg" |
| `mime` | string | Không | Kiểu MIME của hình ảnh | Không có |
| `postname` | string | Không | Metadata bổ sung | Không có |
| `sessionID` | string | Không | Định danh phiên để theo dõi | Không có |
| `deviceInfo` | string | Không | Thông tin thiết bị | Không có |

*Lưu ý: Phải cung cấp `file` hoặc `image_base64`.

#### Ví dụ yêu cầu

**Tải lên File:**
```bash
curl -X POST "http://localhost:8000/ocr-id-card/upload-image-flexible" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/id_card.jpg" \
  -F "sessionID=12345"
```

**Tải lên Base64:**
```bash
curl -X POST "http://localhost:8000/ocr-id-card/upload-image-flexible" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "image_base64=data:image/jpeg;base64,/9j/4AAQSkZJRgABA..." \
  -d "file_name=id_card.jpg" \
  -d "sessionID=12345"
```

#### Định dạng phản hồi

API trả về phản hồi JSON với cấu trúc sau:

```json
{
  "errorCode": "string",
  "errorMessage": "string", 
  "data": [
    {
      // Thông tin thẻ được trích xuất
    }
  ]
}
```

#### Phản hồi thành công (errorCode: "0")

```json
{
  "errorCode": "0",
  "errorMessage": "không có lỗi",
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

#### Phản hồi lỗi

**Mã lỗi "1" - Vấn đề chất lượng/xử lý hình ảnh:**
```json
{
  "errorCode": "1", 
  "errorMessage": "Chất lượng hình ảnh quá thấp hoặc xử lý thất bại",
  "data": []
}
```

**Mã lỗi "2" - Vấn đề phát hiện thẻ:**
```json
{
  "errorCode": "2",
  "errorMessage": "Không phát hiện thẻ trong hình ảnh hoặc loại thẻ không hợp lệ", 
  "data": []
}
```

**Mã lỗi "3" - Vấn đề trích xuất văn bản:**
```json
{
  "errorCode": "3",
  "errorMessage": "Trích xuất văn bản thất bại",
  "data": []
}
```

## Loại thẻ được hỗ trợ

- **CCCD (Căn cước công dân)**: Thẻ căn cước quốc gia Việt Nam
- **CC (Chứng minh nhân dân)**: Thẻ chứng minh nhân dân Việt Nam

## Mặt thẻ được hỗ trợ

- **Mặt trước (Front)**: Chứa thông tin cá nhân
- **Mặt sau (Back)**: Chứa thông tin bổ sung và chữ ký

## Yêu cầu hình ảnh

- **Định dạng**: JPEG, PNG, hoặc các định dạng hình ảnh phổ biến khác
- **Chất lượng**: Khuyến nghị độ phân giải cao để độ chính xác OCR tốt hơn
- **Hướng**: Bất kỳ hướng nào (tự động điều chỉnh)
- **Kích thước**: Kích thước file tối đa phụ thuộc vào cấu hình máy chủ

## Quy trình xử lý

1. **Xác thực hình ảnh**: Kiểm tra chất lượng và định dạng hình ảnh
2. **Phát hiện thẻ**: Phát hiện và cắt thẻ căn cước từ hình ảnh
3. **Phát hiện vùng văn bản**: Xác định các trường văn bản trên thẻ
4. **Trích xuất văn bản**: Trích xuất văn bản sử dụng OCR (VietOCR cho văn bản tiếng Việt, PaddleOCR cho MRZ)
5. **Ánh xạ trường**: Ánh xạ văn bản được trích xuất tới các trường thẻ tương ứng
6. **Tạo phản hồi**: Trả về dữ liệu có cấu trúc

## Hiệu suất

- **Thời gian xử lý trung bình**: 0.2-1 giây mỗi hình ảnh
- **Yêu cầu đồng thời**: Được hỗ trợ với xử lý thread pool
- **Tối ưu hóa bộ nhớ**: Bộ nhớ GPU được tự động xóa sau khi xử lý

## Xử lý lỗi

API bao gồm xử lý lỗi toàn diện cho:
- Định dạng hình ảnh không hợp lệ
- Chất lượng hình ảnh kém
- Thất bại phát hiện thẻ
- Vấn đề trích xuất văn bản
- Lỗi xử lý máy chủ

## Cân nhắc bảo mật

- Hình ảnh được xử lý trong bộ nhớ và không được lưu trữ vĩnh viễn
- Session ID có thể được sử dụng để theo dõi yêu cầu
- Thông tin thiết bị có thể được ghi log cho mục đích kiểm toán

## Giới hạn tốc độ

Giới hạn tốc độ có thể áp dụng tùy thuộc vào cấu hình máy chủ. Liên hệ với quản trị viên để biết giới hạn cụ thể.

## Hỗ trợ

Để được hỗ trợ kỹ thuật hoặc báo cáo vấn đề, vui lòng kiểm tra log máy chủ hoặc liên hệ với nhóm phát triển.
