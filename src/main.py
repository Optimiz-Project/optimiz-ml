import os
import librosa
import numpy as np
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, Response, status
from tensorflow.keras.models import load_model
from typing import Annotated
from src.audio import process_audio, verify_audio_header

# Initialization
load_dotenv()
model_path = os.getenv('MODEL_PATH')
model = load_model(model_path)
app = FastAPI()

@app.get('/')
async def greet():
    results = {
        'status': 200,
        'data': {
            'message': 'Hello, Optimiz!'
        }
    }
    return results

@app.post('/predict-dummy', status_code=201)
async def predict_dummy():
    return {'suara': 'lancar'}

@app.post('/predict', status_code=201)
async def predict(
    req: Request, 
    res: Response,
    file: Annotated[UploadFile, File(description="A file read as UploadFile")]
    ):
    if not verify_audio_header(req.headers.get('Content-Type')):
        res.status_code = status.HTTP_400_BAD_REQUEST
        return {
            'message': 'Invalid file content type.',
            'type': req.headers.get('Content-Type')
        }
    return {
        'headers': req.headers,
        'file': file
    }
