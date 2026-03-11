import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.interface import MnistClassifierInterface


class CNNClassifier(MnistClassifierInterface):
    def __init__(self, image_size=28, num_classes=10):
        self.image_size = image_size
        feature_size = image_size // 4

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)


    def train(self, X_train, y_train, epochs=10, batch_size=64):
        X = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.long)
        X = X.view(-1, 1, self.image_size, self.image_size)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0

            for batch_X, batch_y in tqdm(loader, desc=f"CNN Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"loss: {epoch_loss:.4f}")


    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        X = X.view(-1, 1, self.image_size, self.image_size)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)

        return predicted.numpy()