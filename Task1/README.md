# Task 1 — MNIST Classification

The goal of this task was to implement three classification models for the MNIST handwritten digit dataset using object-oriented design.

## Models

The following models were implemented:

- Random Forest
- Feed-Forward Neural Network
- Convolutional Neural Network (CNN)

All models implement a common interface:
`MnistClassifierInterface` with two methods:
`train(X_train, y_train)`
`predict(X_test)`

A wrapper class `MnistClassifier` selects the model based on the algorithm parameter: rf, nn, and cnn

## Dataset

MNIST dataset (70,000 images, 28×28 grayscale digits).
The dataset is normalized and split into training and test sets.

## Evaluation

Models are evaluated using:

- Confusion Matrix
- Precision
- Recall
- F1-score

Example results:

| Model | Accuracy |
|------|---------|
| Random Forest | ~97% |
| Feed-Forward NN | ~98% |
| CNN | ~99% |

## Run

1. `pip install -r requirements.txt`
2. Open `demo.ipynb` and run all its cells
