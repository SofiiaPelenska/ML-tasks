from models.interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier

class RFClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100, random_state=0):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)