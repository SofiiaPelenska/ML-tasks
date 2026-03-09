from data.mnist_loader import load_mnist
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from classifier import mnist_classifier

def main():
    X_train, X_test, y_train, y_test = load_mnist()

    for key, algo in mnist_classifier.MODELS.items():
        model = mnist_classifier.MnistClassifier(key)

        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        print("-" * 100)
        print(f"Model: {algo.name}")
        print()
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print()

if __name__ == "__main__": 
    main()
