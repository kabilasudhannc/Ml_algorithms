import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    WEIGHTS = 0
    BIAS = 0
    history = {
        'loss': [],
    }

    def random_weights_initializer(self, input_size):
        LinearRegression.WEIGHTS = np.random.randn(1, input_size)
        LinearRegression.BIAS = 0

    def mean_absolute_error(self, y, y_hat):
        mae = np.sum(np.abs(y - y_hat)) / len(y)
        return mae

    def mean_squared_error(self, y, y_hat):
        mse = np.sum((y_hat - y) ** 2) / 2 * len(y)
        return mse

    def gradient_descent(self, x, y, y_hat, learning_rate):
        m = len(y)
        LinearRegression.BIAS = LinearRegression.BIAS - ((learning_rate / m) * np.sum(y_hat - y))
        LinearRegression.WEIGHTS = LinearRegression.WEIGHTS - ((learning_rate / m) * (np.sum((y_hat - y) * x)))

    def fit(self, x, y, learning_rate=0.01, epochs=1000, loss_function='mae'):
        self.random_weights_initializer(x.shape[1])
        for epoch in range(1, epochs + 1):
            y_hat = np.dot(x, LinearRegression.WEIGHTS.T) + LinearRegression.BIAS
            if loss_function == 'mse':
                loss = self.mean_squared_error(y, y_hat)
            else:
                loss = self.mean_absolute_error(y, y_hat)
            print(f'*********** Epoch: {epoch} ***********')
            print(f'{loss_function}: {loss}')
            LinearRegression.history['loss'].append(loss)
            self.gradient_descent(x, y, y_hat, learning_rate)

    def predict(self, x):
        y_hat = np.dot(x, LinearRegression.WEIGHTS.T) + LinearRegression.BIAS
        return y_hat


x = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

y = np.array([
    [3],
    [7],
    [11],
    [15]
])

model = LinearRegression()
model.fit(x, y, epochs=100000, learning_rate=0.001, loss_function='mse')

# plt.subplot()
# plt.plot(LinearRegression.history['loss'][:101])
# plt.show()

x_test = np.array([
    [100, 200],
    [300, 100],
    [400, 500]
])

predicted = model.predict(x_test)
print(predicted)

plt.subplot()
plt.scatter(x=x_test[:, 0], y=x_test[:, 1])
plt.plot(predicted)
plt.show()
