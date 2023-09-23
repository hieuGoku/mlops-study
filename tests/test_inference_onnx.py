import numpy as np
from ml.inference_onnx import Food3ONNXPredictor

def test_onnx_predictor():
    model_path = "ml/weights/onnx/sample.onnx"
    predictor = Food3ONNXPredictor(model_path)
    image = np.random.rand(1, 3, 64, 64).astype(np.float32)
    predictions = predictor.predict(image)
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    test_onnx_predictor()
