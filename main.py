from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.logger import logger
from src.configs import settings
from src.load_resources import app_lifespan
from src.routers import ocr_router
 
logger.info("Starting OCR identification card application...")
app = FastAPI(
    title=settings.app_name,
    description="OCR identification card application",
    version="1.0.0",
    lifespan=app_lifespan
)
 
# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
 
app.include_router(ocr_router.router)
 
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": f"Welcome to {settings.app_name}"}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)
 
 
 
 
 
 
 
 
 
 