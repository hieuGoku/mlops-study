'''This is the main file for the FastAPI app'''

from io import BytesIO
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ml.inference_onnx import Food3ONNXPredictor


app = FastAPI(title="MLOps Basics App")


model_path = "ml/weights/onnx/sample.onnx"
predictor = Food3ONNXPredictor(model_path)


def read_image_request(file):
    """Reads image from request"""
    image = Image.open(BytesIO(file))
    return image


def process_image(image):
    """Preprocesses image for inference"""
    image = image.resize((64, 64))
    image = np.array(image).transpose(2, 0, 1).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.post("/predict/image")
async def get_prediction(file: UploadFile = File(...)):
    image = read_image_request(await file.read())
    image = process_image(image)
    result = predictor.predict(image)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)
