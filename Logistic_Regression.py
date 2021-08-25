import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression:
    WEIGHTS = 0
    BIAS = 0
    history = {
        'loss': [],
        'accuracy': [],
    }

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def accuracy(self, actual, predicted):
        matched = 0
        predicted = (predicted > 0.5).astype(int)
        for y, y_hat in zip(actual, predicted):
            if y == y_hat:
                matched += 1

        return f'{(matched / len(actual)) * 100}%'

    def random_weights_initializer(self, input_size):
        LogisticRegression.WEIGHTS = np.random.randn(1, input_size)
        LogisticRegression.BIAS = 0

    def cross_entropy_loss(self, y, y_hat):
        m = len(y)
        log = (y * np.log(y_hat) + ((1 - y) * np.log(1 - y_hat)))
        loss = -(1 / m) * np.sum(log)
        return loss

    def gradient_descent(self, x, y, y_hat, weights, bias, learning_rate):
        m = len(y)

        LogisticRegression.WEIGHTS = weights - ((learning_rate / m) * np.sum(((y_hat - y) * x)))
        LogisticRegression.BIAS = bias - ((learning_rate / m) * np.sum((y_hat - y)))

    def fit(self, x, y, learning_rate=0.3, epochs=100):
        self.random_weights_initializer(x.shape[1])
        for epochs in range(1, epochs + 1):
            y_hat = np.dot(x, LogisticRegression.WEIGHTS.T) + LogisticRegression.BIAS
            y_hat = self.sigmoid(y_hat)
            LogisticRegression.history['loss'].append(self.cross_entropy_loss(y, y_hat))
            LogisticRegression.history['accuracy'].append(self.accuracy(y, y_hat))
            print(f'\n*********** Epoch: {epochs} ***********')
            print(f"Loss: {self.cross_entropy_loss(y, y_hat) * 100:.2f}%")
            print(f"Accuracy: {self.accuracy(y, y_hat)}")
            self.gradient_descent(x, y, y_hat, LogisticRegression.WEIGHTS, LogisticRegression.BIAS,
                                  learning_rate)

        # plt.subplot()
        # plt.plot(LogisticRegression.LOSS)
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Model Loss')
        # plt.legend(['Training loss'])
        # plt.show()
        #
        # plt.subplot()
        # plt.plot(LogisticRegression.ACCURACY)
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.title('Model Accuracy')
        # plt.legend(['Training Accuracy'])
        # plt.show()

        return LogisticRegression.history

    def predict(self, x):
        p = np.dot(x, LogisticRegression.WEIGHTS.T) + LogisticRegression.BIAS
        return (p > 0.5).astype(int)


x = np.array([
    [.89, .08, .57],
    [.50, 0.72, .67],
    [.44, .99, .42],
    [.22, .30, .38]
])
y = np.array([[1],
              [0],
              [0],
              [1]])

model = LogisticRegression()
history = model.fit(x, y, epochs=10000, learning_rate=0.3)
predictions = model.predict(x)
print(predictions)
print(f'Accuracy: {model.accuracy(y, predictions)}')

plt.subplot()
plt.plot(history['loss'])
plt.show()
