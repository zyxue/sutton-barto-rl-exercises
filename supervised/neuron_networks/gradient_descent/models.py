from tqdm import tqdm
import numpy as np


class LinearModel1D(object):
    def __init__(self, w=None, b=None, learning_rate=0.1, n_epochs=10):
        if w is None:
            self.w = np.random.normal()
        else:
            self.w = w
        if b is None:
            self.b = np.random.normal()
        else:
            self.b = b

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def __str__(self):
        return f'{self.w} * x + {self.b}'

    def __repr__(self):
        return f'{self.__class__.__name__}(w={self.w}, b={self.b}, learning_rate={self.learning_rate}, n_epochs={self.n_epochs}'

    def fit(self, xs, ys, method):
        dd = {
         'bgd': self.batch_gradient_descent,
         'sgd': self.stochastic_gradient_descent
        }
        dd[method](xs, ys)

    def batch_gradient_descent(self, xs, ys):
        self.history = {
            'loss': [self.loss(xs, ys)],
            'w': [self.w],
            'b': [self.b],
        }
        for i in tqdm(range(self.n_epochs)):
            dw, db = self.derivative(xs, ys)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            self.history['loss'].append(self.loss(xs, ys))
            self.history['w'].append(self.w)
            self.history['b'].append(self.w)

    def stochastic_gradient_descent(self, xs, ys):
        pass

    def derivative(self, xs, ys):
        """derivative of rmse"""
        dw = 2 * np.mean((self.w * xs + self.b - ys) * xs)
        db = 2 * np.mean((self.w * xs + self.b - ys))
        return dw, db

    def loss(self, xs, ys):
        return np.mean((self.predict(xs) - ys) ** 2)

    def predict(self, xs):
        return self.w * xs + self.b
