from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path
from src.api.inference import DeepfakeInference
from src.utils.logger import setup_logger

app = FastAPI(title="Deepfake Detection API", version="1.0.0")
logger = setup_logger('API')

# Initialize inference engine
inference_engine = DeepfakeInference()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        inference_engine.load_models()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": inference_engine.models_loaded,
        "device": str(inference_engine.device)
    }

@app.post("/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    """
    Upload video file and get deepfake detection scores
    
    Returns:
    - visual: Visual model prediction and scores
    - audio: Audio model prediction and scores  
    - fusion: Combined model prediction and scores
    """
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are supported")
    
    # Save uploaded file to temporary location
    content = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Run inference with timing
        import time
        start_time = time.time()
        results = inference_engine.predict(tmp_file_path)
        prediction_time = time.time() - start_time
        
        # Add metadata
        response = {
            "filename": file.filename,
            "file_size": len(content),
            "overall_prediction": results['fusion']['prediction'],
            "prediction_time_seconds": round(prediction_time, 3),
            "predictions": results,
            "categories": {
                "A": "Real Video + Real Audio",
                "B": "Real Video + Fake Audio", 
                "C": "Fake Video + Real Audio",
                "D": "Fake Video + Fake Audio"
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except PermissionError:
            logger.warning(f"Could not delete temporary file: {tmp_file_path}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)