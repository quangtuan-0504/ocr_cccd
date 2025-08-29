from ultralytics import YOLO
 
# Load the YOLO11 model
model = YOLO("runs/detect/train_det4cornercard/weights/best.pt")
 
# Export the model to ONNX format
model.export(format="onnx",dynamic=True)  # creates 'yolo11n.onnx'