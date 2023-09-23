'''Inference module for the Food3 classifier.'''

from scipy.special import softmax
import onnxruntime as ort
from ml.utils import timing


class Food3ONNXPredictor:
    '''Predictor class for the Food3 classifier.'''
    def __init__(self, model_path) -> None:
        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.lables = ['pizza', 'steak', 'sushi']

    @timing
    def predict(self, image) -> list:
        '''Predicts the class of an image.'''
        ort_outs = self.ort_session.run(None, {"input": image})
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": str(score)})
        return predictions
