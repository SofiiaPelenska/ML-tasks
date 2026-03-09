from dataclasses import dataclass
from typing import Type

from models.rf_model import RFClassifier
from models.nn_model import NNClassifier
from models.cnn_model import CNNClassifier
from models.interface import MnistClassifierInterface

@dataclass
class AlgorithmType:
    model: Type
    name: str

MODELS = {
    #"rf": AlgorithmType(RFClassifier, "Random Forest"),
    #"nn": AlgorithmType(NNClassifier, "Feed-Forward Neural Network"),
    "cnn": AlgorithmType(CNNClassifier, "Convolutional Neural Network"),
}

class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm: str):
        if (algorithm in MODELS):
            self.model = MODELS[algorithm].model()
        else:
            raise ValueError("Invalid input")
        
    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)