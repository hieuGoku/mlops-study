'''Test inference module.'''

import torch
from ml.inference import Food3Predictor


def test_predictor():
    '''Test predictor.'''
    model_path = "ml/checkpoints/sample.ckpt"
    predictor = Food3Predictor(model_path)
    image = torch.rand(1, 3, 64, 64)
    predictions = predictor.predict(image)
    assert len(predictions) == 3
    assert isinstance(predictions[0], dict)
    assert "label" in predictions[0]
    assert "score" in predictions[0]
    print(f"[INFO] Predictions: {predictions}")

if __name__ == "__main__":
    test_predictor()
