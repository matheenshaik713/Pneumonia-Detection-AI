"""
FastAPI Backend

Author: Matheen Shaik
"""

from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.predict import Predictor

app = FastAPI(
    title="Pneumonia Detection API",
    version="1.0"
)

predictor = Predictor()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():

    return {
        "message": "Pneumonia Detection API is Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    filepath = UPLOAD_DIR / file.filename

    with open(filepath, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = predictor.predict(str(filepath))

    return JSONResponse(
    {
        "filename": file.filename,
        "prediction": str(prediction),
        "confidence": float(round(float(confidence), 2))
    }
)