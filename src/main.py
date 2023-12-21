import os
import librosa
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from tensorflow.keras.models import load_model

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

@app.post('/predict')
async def predict(request):
    results = {
        'status': 200,
        'data': {
            'message': 'Hello, Optimiz!'
        }
    }
    return results
