import onnxruntime as ort
import cv2
import numpy as np



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
def rescale_back(results,img_w,img_h):
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
    keep, keep_confidences = NMS(boxes,confidence)
    print(np.array(keep).shape)
    return keep, keep_confidences
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (dw, dh)



mode_path = "model/det4corner/best.onnx"
onnx_model = ort.InferenceSession(mode_path)

classes = {
    0: 'top_left',
    1: 'top_right',
    2: 'bottom_right',
    3: 'bottom_left',
    4: 'cc_front',
    5: 'cc_back',
    6: "cccd_front",
    7: "cccd_back",
    8: "other_card"
}


image_ori = cv2.imread("analysis_error/crop/EKYC_Android_20250517_031847.047_UQX_ofl.jpg")
image=image_ori.copy()
# Original image dimensions
orig_h, orig_w = image.shape[:2]

# Preprocess with letterbox (like Ultralytics)
img, ratio, pad = letterbox(image, new_shape=(640, 640))

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize and transpose
img = img.transpose(2, 0, 1)  # HWC to CHW
img = img[np.newaxis, :, :, :]  # Add batch dimension
img = img.astype(np.float32) / 255.0

# Get input name
input_name = onnx_model.get_inputs()[0].name

# Run inference
outputs = onnx_model.run(None, {input_name: img})

results = outputs[0]
results = results.transpose()

results = filter_Detections(results)
print(results)

rescaled_results, confidences = rescale_back(results, orig_w, orig_h)

for res, conf in zip(rescaled_results, confidences):

    x1,y1,x2,y2, cls_id = res
    cls_id = int(cls_id)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    conf = "{:.2f}".format(conf)
    # draw the bounding boxes
    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0, 0),1)
    cv2.putText(image,classes[cls_id]+' '+conf,(x1,y1-17),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)


cv2.imwrite("Output.jpg", image)