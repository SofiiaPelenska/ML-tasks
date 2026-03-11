import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.interface import MnistClassifierInterface


class NNClassifier(MnistClassifierInterface):
    def __init__(self, input_size=784, num_classes=10):
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)


    def train(self, X_train, y_train, epochs=50, batch_size=64):
        X = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0

            for batch_X, batch_y in tqdm(loader, desc=f"NN Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"loss: {epoch_loss:.4f}")


    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)

        return predicted.numpy()