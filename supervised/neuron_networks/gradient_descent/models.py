from tqdm import tqdm
import numpy as np


class BaseModel(object):
    """only consider model with two parameters (w1, w2) for illustration purpose"""
    def __init__(self, w=None, learning_rate=0.1, n_epochs=10):
        if w is None:
            self.w = np.random.normal(size=2)
        else:
            self.w = w

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def __repr__(self):
        return f'{self.__class__.__name__}(w={self.w}, learning_rate={self.learning_rate}, n_epochs={self.n_epochs}'

    def fit(self, xs, ys, method, **kw):
        if method == 'bgd':
            self.batch_gradient_descent(xs, ys)
        elif method == 'sgd':
            self.stochastic_gradient_descent(xs, ys)
        elif method == 'momentum':
            self.momentum(xs, ys)

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

    def momentum(self, xs, ys, gamma=0.9):
        self.init_history(xs, ys)

        v_prev = np.zeros(self.w.shape)
        for i in tqdm(range(self.n_epochs)):
            for x, y in zip(xs, ys):
                x = np.array([x])
                y = np.array([y])

                dw = self.derivative(x, y)
                v_curr = gamma * v_prev + self.learning_rate * dw
                self.w -= v_curr

                self.update_history(x, y)

    def nesterov(self, xs, ys, gamma=0.9):
        pass

    def init_history(self, xs, ys):
        self.history = {
            'loss': [self.loss(xs, ys)],
            'w': [self.w],
        }

    def update_params(self, xs, ys):
        dw = self.derivative(xs, ys)
        self.w = self.w - self.learning_rate * dw

    def update_history(self, xs, ys):
        self.history['loss'].append(self.loss(xs, ys))
        self.history['w'].append(self.w)

    def loss(self, xs, ys):
        """rmse"""
        return np.mean((self.predict(xs) - ys) ** 2)

    def predict(self, xs):
        return NotImplementedError

    def derivative(self, xs, ys):
        raise NotImplementedError


# class LinearModel1D(BaseModel):
#     def predict(self, xs):
#         return self.w * xs + self.b

#     def derivative(self, xs, ys):
#         """derivative of rmse"""
#         dw1 = np.mean((self.w1 * xs + self.w2 - ys) * xs)
#         dw2 = np.mean((self.w1 * xs + self.w2 - ys))
#         return dw1, dw2


class QuadraticModel(BaseModel):
    def predict(self, xs):
        return np.dot(xs ** 2, self.w)

    def derivative(self, xs, ys):
        p = self.predict(xs) - ys  # the common part
        m = p.shape
        dw = np.dot(p, xs ** 2) / m
        return dw

    def derivative_nesterov(self, xs, ys, gamma, v_prev):
        w1 = self.w1 - v_pe

        p = self.predict(xs) - ys  # the common part
        dw1 = np.mean(p * xs[:, 0] ** 2)  # omit a factor of 2, a constant
        dw2 = np.mean(p * xs[:, 1] ** 2)  # omit a factor of 2, a constant
        return dw1, dw2
