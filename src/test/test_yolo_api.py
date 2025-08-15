from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import torch

app = FastAPI(title="YOLOv11 Object Detection API")

# Load YOLOv11 model (using pre-trained weights, e.g., yolov11n)
model = YOLO("model/detcls_text_region/best.pt")

# Ensure model is on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Run YOLOv11 inference with no_grad to save memory
        with torch.no_grad():
            results = model(image_np)
        
        # Extract predictions
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                pred = {
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": [float(coord) for coord in box.xyxy[0].tolist()]
                }
                predictions.append(pred)
        
        # Clean up tensors and variables
        del image_np, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return JSONResponse(content={"predictions": predictions})
    
    except Exception as e:
        # Clean up in case of error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)