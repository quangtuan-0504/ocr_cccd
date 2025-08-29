import json
import os
import shutil
from pathlib import Path
import random
from PIL import Image
import math

def convert_rectangle_to_yolo(points, img_width, img_height):
    """
    Convert rectangle points to YOLOv11 format (normalized coordinates)
    YOLOv11 format: x_center y_center width height (normalized)
    """
    # points có 2 điểm: [x1, y1] và [x2, y2]
    x1, y1 = points[0]
    x2, y2 = points[1]

    # Tính center, width, height
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # Normalize theo kích thước ảnh
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm

def create_yolo_annotation(json_file, img_file, class_mapping):
    """
    Create YOLOv11 annotation from JSON file
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get image dimensions
    img = Image.open(img_file)
    img_width, img_height = img.size

    annotations = []

    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            label = shape['label']
            points = shape['points']

            # Get class ID from mapping
            if label in class_mapping:
                class_id = class_mapping[label]

                # Convert rectangle to YOLO format
                x_center, y_center, width, height = convert_rectangle_to_yolo(points, img_width, img_height)

                # Create annotation line: class_id x_center y_center width height
                annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                annotations.append(annotation)

    return annotations

def convert_to_yolov11_detection(data_dir, output_dir, train_ratio=0.8, class_mapping=None):
    """
    Convert data to YOLOv11 Object Detection format and split into train/val
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Create output directories
    train_images_dir = output_path / "train" / "images"
    train_labels_dir = output_path / "train" / "labels"
    val_images_dir = output_path / "val" / "images"
    val_labels_dir = output_path / "val" / "labels"

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(data_path.glob(ext))

    # Create default class mapping if not provided
    if class_mapping is None:
        # Scan all JSON files to find unique labels
        all_labels = set()
        for img_file in image_files:
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for shape in data['shapes']:
                    if shape['shape_type'] == 'rectangle':
                        all_labels.add(shape['label'])

        # Create class mapping
        class_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
        print(f"Auto-detected classes: {class_mapping}")

    # Shuffle and split data
    random.seed(42)  # For reproducible results
    random.shuffle(image_files)
    train_size = int(len(image_files) * train_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    def process_files(files, images_dir, labels_dir, split_name):
        processed_count = 0
        for img_file in files:
            json_file = img_file.with_suffix('.json')

            if json_file.exists():
                try:
                    # Create annotation
                    annotations = create_yolo_annotation(json_file, img_file, class_mapping)

                    if annotations:  # Only process if there are annotations
                        # Copy image file
                        dst_img = images_dir / img_file.name
                        shutil.copy2(img_file, dst_img)

                        # Create label file
                        label_file = labels_dir / f"{img_file.stem}.txt"
                        with open(label_file, 'w') as f:
                            f.write('\n'.join(annotations))

                        processed_count += 1
                    else:
                        print(f"No valid annotations found in {json_file}")

                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
            else:
                print(f"JSON file not found for {img_file}")

        print(f"Processed {processed_count} files for {split_name}")
        return processed_count

    # Process train and val splits
    train_count = process_files(train_files, train_images_dir, train_labels_dir, "train")
    val_count = process_files(val_files, val_images_dir, val_labels_dir, "val")

    # Create classes.txt file
    classes_file = output_path / "classes.txt"
    with open(classes_file, 'w') as f:
        for label in sorted(class_mapping.keys()):
            f.write(f"{label}\n")

    # Create dataset config file for YOLOv11
    config_content = f"""# YOLOv11 Object Detection Dataset Config
path: {output_path.absolute()}
train: train/images
val: val/images

# Number of classes
nc: {len(class_mapping)}

# Classes
names:
"""
    for label in sorted(class_mapping.keys()):
        config_content += f"  {class_mapping[label]}: {label}\n"

    config_file = output_path / "dataset.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"\n{'='*50}")
    print(f"CONVERSION COMPLETED!")
    print(f"{'='*50}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Total classes: {len(class_mapping)}")
    print(f"Config file: {config_file}")
    print(f"Classes file: {classes_file}")

    return class_mapping

def validate_dataset(output_dir):
    """
    Validate the created dataset
    """
    output_path = Path(output_dir)

    print(f"\n{'='*50}")
    print(f"DATASET VALIDATION")
    print(f"{'='*50}")

    # Check directories
    train_images = output_path / "train" / "images"
    train_labels = output_path / "train" / "labels"
    val_images = output_path / "val" / "images"
    val_labels = output_path / "val" / "labels"

    train_img_count = len(list(train_images.glob("*.*")))
    train_label_count = len(list(train_labels.glob("*.txt")))
    val_img_count = len(list(val_images.glob("*.*")))
    val_label_count = len(list(val_labels.glob("*.txt")))

    print(f"Train: {train_img_count} images, {train_label_count} labels")
    print(f"Val: {val_img_count} images, {val_label_count} labels")

    # Check if image and label counts match
    if train_img_count == train_label_count and val_img_count == val_label_count:
        print("✅ Image and label counts match!")
    else:
        print("❌ Image and label counts don't match!")

    # Sample a few label files to check format
    sample_labels = list(train_labels.glob("*.txt"))[:3]
    print(f"\nSample annotations:")
    for label_file in sample_labels:
        print(f"  {label_file.name}:")
        with open(label_file, 'r') as f:
            lines = f.readlines()[:3]  # First 3 lines
            for line in lines:
                print(f"    {line.strip()}")

# Example usage
if __name__ == "__main__":
    # Cấu hình
    DATA_DIR = "data/v2/datalabeled_for_det4cornercard"  # Thư mục chứa dữ liệu gốc
    OUTPUT_DIR = "data/v2/dataset_yolov11_det4cornercard"  # Thư mục output
    TRAIN_RATIO = 0.8  # Tỷ lệ train (80% train, 20% val)

    # Mapping tùy chỉnh class (tùy chọn)
    # Nếu không cung cấp, sẽ tự động tạo từ các label trong dữ liệu
    CLASS_MAPPING = {
        'top_left': 0,
        'top_right': 1,
        'bottom_right': 2,
        'bottom_left': 3,
        'cc_front': 4,
        'cc_back': 5,
        "cccd_front": 6,
        "cccd_back": 7,
        "other_card": 8
    }

    # Thực hiện convert
    class_mapping = convert_to_yolov11_detection(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        class_mapping=CLASS_MAPPING  # Có thể set None để tự động detect
    )

    print(f"\nFinal class mapping:")
    for label, class_id in class_mapping.items():
        print(f"  {class_id}: {label}")

    # Validate dataset
    validate_dataset(OUTPUT_DIR)

    print(f"\n{'='*50}")
    print(f"READY TO TRAIN!")
    print(f"{'='*50}")
    print(f"Use this command to train YOLOv11:")
    print(f"yolo detect train data={OUTPUT_DIR}/dataset.yaml model=yolo11n.pt epochs=100")