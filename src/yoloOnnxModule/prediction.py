import cv2
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import onnxruntime

def preprocess_img(image: np.ndarray, imgsz: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess the input image for YOLO Onnx model inference.

    Args:
        image: Input image as a NumPy array.
        imgsz: Target image size as a tuple (width, height).

    Returns:
        Preprocessed image as a NumPy array.
    """
    img = cv2.resize(image, imgsz)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, *imgsz)
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Convert image to float32
    img = img.astype(np.float32)

    return img

def filter_Detections(results, thresh = 0.5):
    # if model is trained on 1 class only
    if len(results[0]) == 5:
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in results if detection[4] > thresh]
        considerable_detections = np.array(considerable_detections)
        return considerable_detections

    # if model is trained on multiple classes
    else:
        A = []
        for detection in results:

            class_id = detection[4:].argmax()
            confidence_score = detection[4:].max()

            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)

        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)

        return considerable_detections

def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences



# function to rescale bounding boxes
def rescale_back(results,img_w,img_h, iou_thresh):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence, iou_thresh)
    # print(np.array(keep).shape)
    return keep, keep_confidences

def yoloonnx_predict(image: np.ndarray, model: onnxruntime.InferenceSession, format_box: str, configs: Dict[str, Any]) -> np.ndarray:
    """
    Run inference on the input image using the YOLO Onnx model.

    Args:
        image: Input image as a NumPy array.
        model: Loaded YOLO Onnx model instance.

    Returns:
        Detections as a NumPy array.
    """
    conf = configs.get("conf", 0.25)
    imgsz = configs["imgsz"]
    iou = configs.get("iou", 0.45)

    img_w, img_h = image.shape[1], image.shape[0]
    preprocessed_image = preprocess_img(image, imgsz)
    outputs = model.run(None, {"images": preprocessed_image})
    results = outputs[0]
    results = results.transpose()
    results = filter_Detections(results, thresh=conf)
    rescaled_results, confidences = rescale_back(results, img_w, img_h, iou)

    rescaled_results = np.array(rescaled_results)
    conf = np.array(confidences)

    cls = rescaled_results[:, 4].astype(int)
    xyxy = rescaled_results[:, :4]
    if format_box == 'xyxy':
        del rescaled_results, preprocessed_image
        return xyxy, conf, cls
    else:
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        xywh = np.column_stack((x_center, y_center, w, h))
        # if configs['save']:
        #     for res, conf in zip(rescaled_results, confidences):
        #         x1,y1,x2,y2, cls_id = res
        #         cls_id = int(cls_id)
        #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #         conf = "{:.2f}".format(conf)
        #         # draw the bounding boxes
        #         cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0, 0),1)
        #         cv2.putText(image,classes[cls_id]+' '+conf,(x1,y1-17),
        #                     cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)
        #     cv2.imwrite("Output.jpg", image)
        del rescaled_results, xyxy, x_center, y_center, w, h, preprocessed_image
        return xywh, conf, cls
