#!/usr/bin/env python3
"""
Start the Deepfake Detection API server
"""

import uvicorn
from src.api.main import app

if __name__ == "__main__":
    print("Starting Deepfake Detection API...")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False
    )