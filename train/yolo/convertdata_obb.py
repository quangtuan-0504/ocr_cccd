import json
import os
import shutil
from pathlib import Path
import random
from PIL import Image
import math

def convert_polygon_to_obb(points, img_width, img_height):
    """
    Convert polygon points to YOLOv8 OBB format (normalized coordinates)
    YOLOv8 OBB format: x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
    """
    # Normalize coordinates
    normalized_points = []
    for point in points:
        x_norm = point[0] / img_width
        y_norm = point[1] / img_height
        normalized_points.extend([x_norm, y_norm])

    return normalized_points

def create_yolo_annotation(json_file, img_file, class_mapping):
    """
    Create YOLOv8 OBB annotation from JSON file
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get image dimensions
    img = Image.open(img_file)
    img_width, img_height = img.size

    annotations = []

    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            label = shape['label']
            points = shape['points']

            # Get class ID from mapping
            if label in class_mapping:
                class_id = class_mapping[label]

                # Convert polygon to OBB format
                obb_coords = convert_polygon_to_obb(points, img_width, img_height)

                # Create annotation line: class_id x1 y1 x2 y2 x3 y3 x4 y4
                annotation = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in obb_coords)
                annotations.append(annotation)

    return annotations

def convert_to_yolov8_obb(data_dir, output_dir, train_ratio=0.8, class_mapping=None):
    """
    Convert data to YOLOv8 OBB format and split into train/val
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
                    if shape['shape_type'] == 'polygon':
                        all_labels.add(shape['label'])

        # Create class mapping
        class_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
        print(f"Auto-detected classes: {class_mapping}")

    # Shuffle and split data
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

                except Exception as e:
                    print(f"Error processing {img_file}: {e}")

        print(f"Processed {processed_count} files for {split_name}")

    # Process train and val splits
    process_files(train_files, train_images_dir, train_labels_dir, "train")
    process_files(val_files, val_images_dir, val_labels_dir, "val")

    # Create classes.txt file
    classes_file = output_path / "classes.txt"
    with open(classes_file, 'w') as f:
        for label in sorted(class_mapping.keys()):
            f.write(f"{label}\n")

    # Create dataset config file for YOLOv8
    config_content = f"""# YOLOv8 OBB Dataset Config
path: {output_path.absolute()}
train: train/images
val: val/images

nc: {len(class_mapping.keys())}
# Classes
names:
"""
    for label in sorted(class_mapping.keys()):
        config_content += f"  {class_mapping[label]}: {label}\n"

    config_file = output_path / "dataset.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"\nConversion completed!")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    print(f"Classes: {len(class_mapping)}")
    print(f"Config file: {config_file}")

    return class_mapping

# Example usage
if __name__ == "__main__":
    # Cấu hình
    DATA_DIR = "/home/tuanndq/ocr_cccd/yolo/data/dataset_obb_v1_labeled"  # Thư mục chứa dữ liệu gốc
    OUTPUT_DIR = "./data/yolov8_obb_dataset"  # Thư mục output
    TRAIN_RATIO = 0.8  # Tỷ lệ train (80% train, 20% val)

    # Mapping tùy chỉnh class (tùy chọn)
    # Nếu không cung cấp, sẽ tự động tạo từ các label trong dữ liệu
    CLASS_MAPPING = {
        "cccd_front": 0,
        "cccd_back": 1,
        "cc_front": 2,
        "cc_back": 3,
        "other_card": 4 
    }

    # Thực hiện convert
    class_mapping = convert_to_yolov8_obb(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        class_mapping=CLASS_MAPPING  # Có thể set None để tự động detect
    )

    print("\nClass mapping used:")
    for label, class_id in class_mapping.items():
        print(f"  {label}: {class_id}")