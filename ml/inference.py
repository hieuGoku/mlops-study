'''Inference module for the Food3 classifier.'''

import torch
from ml.data_module import Food3DataModule
from ml.lightning_module import Food3LM
from ml.utils import timing


class Food3Predictor:
    '''Predictor class for the Food3 classifier.'''
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.model = Food3LM.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = Food3DataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.lables = ['pizza', 'steak', 'sushi']

    @timing
    def predict(self, image) -> list:
        '''Predicts the class of an image.'''
        logits = self.model(image)
        scores = self.softmax(logits).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions
