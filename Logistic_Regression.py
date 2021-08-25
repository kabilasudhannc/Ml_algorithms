import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def accuracy(actual, predicted):
    matched = 0
    predicted = (predicted > 0.5).astype(int)
    for y, y_hat in zip(actual, predicted):
        if y == y_hat:
            matched += 1

    return matched / len(actual)


def random_weights_initializer(input_size):
    weights = np.random.randn(1, input_size)
    bias = 0

    return weights, bias


def cross_entropy_loss(y, y_hat):
    m = len(y)
    log = (y * np.log(y_hat) + ((1 - y) * np.log(1 - y_hat)))
    loss = -(1 / m) * np.sum(log)
    return loss


def gradient_descent(x, y, y_hat, weights, bias, learning_rate):
    m = len(y)

    weights = weights - ((learning_rate / m) * np.sum(((y_hat - y) * x)))
    bias = bias - ((learning_rate / m) * np.sum((y_hat - y)))

    return weights, bias


def model(x, y, learning_rate=0.3, epochs=100):
    weights, bias = random_weights_initializer(x.shape[1])
    loss = []
    acc = []
    for epochs in range(epochs):
        y_hat = np.dot(x, weights.T) + bias
        y_hat = sigmoid(y_hat)
        loss.append(cross_entropy_loss(y, y_hat))
        acc.append(accuracy(y, y_hat))
        print(f'\n*********** Epoch: {epochs} ***********')
        print(f"Loss: {cross_entropy_loss(y, y_hat) * 100:.2f}%")
        print(f"Accuracy: {accuracy(y, y_hat) * 100}%")
        weights, bias = gradient_descent(x, y, y_hat, weights, bias, learning_rate)

    plt.subplot()
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend(['Training loss'])
    plt.show()

    plt.subplot()
    plt.plot(acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend(['Training Accuracy'])
    plt.show()

    return weights, bias


x = np.array([
    [.89, .08, .57],
    [.50, 0.72, .67],
    [.44, .99, .42],
    [.22, .30, .38]
])
y = np.array([[0],
              [1],
              [1],
              [0]])
print(x)
print(y)
model(x, y, epochs=20000)
