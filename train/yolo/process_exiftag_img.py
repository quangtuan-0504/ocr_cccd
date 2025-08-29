import os
from PIL import Image, ExifTags

def correct_image_orientation(image_path):
    try:
        image = Image.open(image_path)
        # Kiểm tra metadata EXIF để lấy thông tin hướng
        for orientation_key in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation_key] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation_key, 1)
            # Xoay ảnh dựa trên giá trị orientation
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
            # Ghi đè ảnh đã xoay lên file gốc
            image.save(image_path, quality=95)  # quality=95 để giữ chất lượng ảnh
            print(f"Corrected orientation for {image_path}")
        else:
            print(f"No EXIF data found for {image_path}, no changes made")
    except (AttributeError, KeyError, IndexError) as e:
        print(f"Error processing {image_path}: {e}, no changes made")

def process_images_in_folder(folder_path):
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {image_path}")
            correct_image_orientation(image_path)

# Thư mục chứa ảnh
folder_path = "data/v2/datalabeled_for_det4cornercard"
# folder_path = "data/dataset_obb_v1_labeled"

process_images_in_folder(folder_path)