from tqdm import tqdm
import numpy as np


class BaseModel(object):
    """only consider model with two parameters (w1, w2) for illustration purpose"""
    def __init__(self, w1=None, w2=None, learning_rate=0.1, n_epochs=10):
        if w1 is None:
            self.w1 = np.random.normal()
        else:
            self.w1 = w1
        if w2 is None:
            self.w2 = np.random.normal()
        else:
            self.w2 = w2

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def __repr__(self):
        return f'{self.__class__.__name__}(w={self.w1}, b={self.w2}, learning_rate={self.learning_rate}, n_epochs={self.n_epochs}'

    def fit(self, xs, ys, method):
        dd = {
            'bgd': self.batch_gradient_descent,
            'sgd': self.stochastic_gradient_descent,
        }
        dd[method](xs, ys)

    def batch_gradient_descent(self, xs, ys):
        self.init_history(xs, ys)
        for i in tqdm(range(self.n_epochs)):
            self.update_params(xs, ys)
            self.update_history(xs, ys)

    def stochastic_gradient_descent(self, xs, ys):
        self.init_history(xs, ys)
        for i in tqdm(range(self.n_epochs)):
            for x, y in zip(xs, ys):
                x = np.array([x])
                y = np.array([y])
                self.update_params(x, y)
                self.update_history(x, y)

    def init_history(self, xs, ys):
        self.history = {
            'loss': [self.loss(xs, ys)],
            'w1': [self.w1],
            'w2': [self.w2],
        }

    def update_params(self, xs, ys):
        dw1, dw2 = self.derivative(xs, ys)
        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2

    def update_history(self, xs, ys):
        self.history['loss'].append(self.loss(xs, ys))
        self.history['w1'].append(self.w1)
        self.history['w2'].append(self.w2)

    def loss(self, xs, ys):
        """rmse"""
        return np.mean((self.predict(xs) - ys) ** 2)

    def predict(self, xs):
        return NotImplementedError

    def derivative(self, xs, ys):
        raise NotImplementedError


class LinearModel1D(BaseModel):
    def predict(self, xs):
        return self.w * xs + self.b

    def derivative(self, xs, ys):
        """derivative of rmse"""
        dw1 = np.mean((self.w1 * xs + self.w2 - ys) * xs)
        dw2 = np.mean((self.w1 * xs + self.w2 - ys))
        return dw1, dw2


class QuadraticModel(BaseModel):
    def predict(self, xs):
        return self.w1 * xs[:, 0] ** 2 + self.w2 * xs[:, 1] ** 2

    def derivative(self, xs, ys):
        p = self.predict(xs) - ys  # the common part
        dw1 = np.mean(p * xs[:, 0] ** 2)  # omit a factor of 2, a constant
        dw2 = np.mean(p * xs[:, 1] ** 2)  # omit a factor of 2, a constant
        return dw1, dw2
