import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Union


class YOLOv11OnnxRuntime:
    """
    YOLOv11 object detection model class for handling ONNX inference and visualization.

    This class provides functionality to perform inference on images,
    and visualize the detection results with bounding boxes and labels.

    Attributes:
        onnx_model (ort.InferenceSession): Instance of the ONNX model file.
        classes (List[str]): List of class names from the COCO dataset.
        color_palette (np.ndarray): Random color palette for visualizing different classes.
        input_width (int): Width dimension of the model input.
        input_height (int): Height dimension of the model input.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.

    Methods:
        letterbox: Resize and reshape images while maintaining aspect ratio by adding padding.
        draw_detections: Draw bounding boxes and labels on the input image based on detected objects.
        preprocess: Preprocess the input image before performing inference.
        postprocess: Perform post-processing on the model's output to extract and visualize detections.
        predict: Perform inference using an ONNX model and return the output with detections.
        convert_boxes: Convert boxes between different formats (xyxy, xywh).
    """

    def __init__(self, onnx_model: ort.InferenceSession):
        """
        Initialize an instance of the YOLOv11OnnxRuntime class.

        Args:
            onnx_model (ort.InferenceSession): Instance of the ONNX model.
        """
        self.onnx_model = onnx_model
        self.classes = []
        self.color_palette = None

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            gain (float): Scale factor used for resizing.
            pad (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        gain = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * gain)), int(round(shape[0] * gain))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, gain, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        """Draw bounding boxes and labels on the input image based on the detected objects."""
        # Extract the coordinates of the bounding box (xyxy format)
        x1, y1, x2, y2 = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id].tolist()

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = int(x1)
        label_y = int(y1) - 10 if int(y1) - 10 > label_height else int(y1) + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess the input image before performing inference.

        Args:
            img (np.ndarray): Input image to preprocess.

        Returns:
            image_data (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            gain (float): Scale factor used during letterboxing.
            pad (Tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        # Get the height and width of the input image
        self.img_height, self.img_width = img.shape[:2]

        # Convert the image color space from BGR to RGB
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img.copy()
        # Apply letterboxing
        img_resized, gain, pad = self.letterbox(img_rgb, (self.input_height, self.input_width))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img_resized) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data, gain, pad

    def convert_xyxy_to_xywh_center(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert boxes from xyxy format to xywh format with center coordinates.

        Args:
            boxes (np.ndarray): Input boxes in xyxy format.

        Returns:
            np.ndarray: Boxes in xywh format (x_center, y_center, width, height).
        """
        converted_boxes = np.zeros_like(boxes)
        converted_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
        converted_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
        converted_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]        # width
        converted_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]        # height
        return converted_boxes

    def convert_xyxy_to_xywh_topleft(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert xyxy boxes to xywh format with top-left corner (for cv2.dnn.NMSBoxes).

        Args:
            boxes (np.ndarray): Boxes in xyxy format.

        Returns:
            np.ndarray: Boxes in xywh format with top-left corner.
        """
        boxes_xywh = np.zeros_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0]  # x (top-left)
        boxes_xywh[:, 1] = boxes[:, 1]  # y (top-left)
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        return boxes_xywh

    def scale_boxes_to_original(self, boxes: np.ndarray, gain: float, pad: Tuple[int, int], format_type: str = "xyxy") -> np.ndarray:
        """
        Scale boxes from model input size to original image size.

        Args:
            boxes (np.ndarray): Boxes in model input coordinates.
            gain (float): Scale factor used during preprocessing.
            pad (Tuple[int, int]): Padding values (top, left) applied during preprocessing.
            format_type (str): Box format ('xyxy' or 'xywh').

        Returns:
            np.ndarray: Boxes scaled to original image coordinates.
        """
        if len(boxes) == 0:
            return boxes

        scaled_boxes = boxes.copy().astype(float)
        pad_top, pad_left = pad

        if format_type == "xyxy":
            # Remove padding and scale back
            scaled_boxes[:, [0, 2]] = (scaled_boxes[:, [0, 2]] - pad_left) / gain  # x coordinates
            scaled_boxes[:, [1, 3]] = (scaled_boxes[:, [1, 3]] - pad_top) / gain   # y coordinates
        else:  # xywh
            # Remove padding from center coordinates and scale back
            scaled_boxes[:, 0] = (scaled_boxes[:, 0] - pad_left) / gain  # x center
            scaled_boxes[:, 1] = (scaled_boxes[:, 1] - pad_top) / gain   # y center
            scaled_boxes[:, 2] = scaled_boxes[:, 2] / gain               # width
            scaled_boxes[:, 3] = scaled_boxes[:, 3] / gain               # height

        # Clip to image boundaries
        if format_type == "xyxy":
            scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, self.img_width)
            scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, self.img_height)
        else:  # xywh
            scaled_boxes[:, 0] = np.clip(scaled_boxes[:, 0], 0, self.img_width)
            scaled_boxes[:, 1] = np.clip(scaled_boxes[:, 1], 0, self.img_height)

        return scaled_boxes

    def postprocess(
        self,
        output: List[np.ndarray],
        gain: float,
        pad: Tuple[int, int],
        conf: float,
        iou: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform post-processing on the model's output to extract detections.

        Args:
            output (List[np.ndarray]): The output arrays from the model.
            gain (float): Scale factor used during preprocessing.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.
            conf (float): Confidence threshold.
            iou (float): IoU threshold for NMS.

        Returns:
            Tuple[np.ndarray]: Boxes (xyxy format in model coordinates), scores, and class IDs.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= conf:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row (center format)
                x_center, y_center, w, h = outputs[i][:4]

                # Convert to xyxy format
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2

                # Add the detection to the lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([x1, y1, x2, y2])

        if not boxes:
            return np.array([]), np.array([]), np.array([])

        # Convert to numpy arrays
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        # Convert xyxy boxes to xywh format with top-left corner for NMS
        boxes_xywh_nms = self.convert_xyxy_to_xywh_topleft(boxes)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh_nms.tolist(), 
            scores.tolist(), 
            conf, 
            iou
        )

        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        # Extract the selected detections
        selected_indices = indices.flatten()
        final_boxes = boxes[selected_indices]
        final_scores = scores[selected_indices]
        final_class_ids = class_ids[selected_indices]

        return final_boxes, final_scores, final_class_ids

    def predict(self, image: np.ndarray, configs: Dict[str, Union[str, int, float, bool, List, Tuple]]) -> Dict[str, np.ndarray]:
        """
        Perform inference using an ONNX model and return comprehensive detection results.

        Args:
            image (np.ndarray): Input image for inference.
            configs (Dict): Configuration parameters including conf, iou, imgsz, save, etc.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'boxes_xyxy_original': Bounding boxes in xyxy format (original image coordinates)
                - 'boxes_xywh_original': Bounding boxes in xywh format (original image coordinates)
                - 'scores': Confidence scores
                - 'class_ids': Class IDs
                - 'annotated_image': Image with drawn detections (optional)
        """
        # Extract configuration parameters
        conf = configs.get("conf", 0.25)
        iou = configs.get("iou", 0.45)
        imgsz = configs.get("imgsz", (640, 640))
        save = configs.get("save", False)
        save_path = os.path.join(
            configs.get('project', 'runs'), 
            configs.get('name', 'detect'), 
            'image.jpg'
        )
        classes = configs.get("classes", [])

        # Setup model parameters
        self.input_height, self.input_width = imgsz
        self.classes = classes
        
        # Generate color palette if not exists or classes changed
        if self.color_palette is None or len(self.color_palette) != len(self.classes):
            self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Get model inputs
        model_inputs = self.onnx_model.get_inputs()

        # Preprocess the image
        img_data, gain, pad = self.preprocess(image)

        # Run inference
        outputs = self.onnx_model.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing (returns boxes in model coordinates, xyxy format)
        boxes_model_xyxy, scores, class_ids = self.postprocess(outputs, gain, pad, conf, iou)

        # Prepare results dictionary
        results = {
            'boxes_xyxy_original': np.array([]),
            'boxes_xywh_original': np.array([]),
            'scores': scores,
            'class_ids': class_ids
        }

        if len(boxes_model_xyxy) > 0:
            # Scale boxes to original image coordinates
            boxes_xyxy_original = self.scale_boxes_to_original(boxes_model_xyxy, gain, pad, "xyxy")
            # Convert to xywh format with center coordinates
            boxes_xywh_original = self.convert_xyxy_to_xywh_center(boxes_xyxy_original)
            
            results['boxes_xyxy_original'] = boxes_xyxy_original
            results['boxes_xywh_original'] = boxes_xywh_original

            # Save image if requested
            if save:
                annotated_image = image.copy()
                for box, score, class_id in zip(boxes_xyxy_original, scores, class_ids):
                    self.draw_detections(annotated_image, box, score, class_id)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, annotated_image)

        return results
