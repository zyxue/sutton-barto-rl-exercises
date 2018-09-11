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
        self.init_history(xs, ys)
        self.update_history(xs, ys)

        if method == 'bgd':
            self.batch_gradient_descent(xs, ys)
        elif method == 'sgd':
            self.stochastic_gradient_descent(xs, ys)
        elif method == 'momentum':
            self.momentum(xs, ys)
        elif method == 'nesterov':
            self.momentum(xs, ys, nesterov=True)
        else:
            raise NotImplementedError(method)

        self.history['w'] = np.array(self.history['w'])

    def batch_gradient_descent(self, xs, ys):
        for i in tqdm(range(self.n_epochs)):
            self.update_params(xs, ys)
            self.update_history(xs, ys)

    def stochastic_gradient_descent(self, xs, ys):
        for i in tqdm(range(self.n_epochs)):
            for _x, _y in zip(xs, ys):
                _x = np.array([_x])
                _y = np.array([_y])
                self.update_params(_x, _y)
                self.update_history(xs, ys)

    def momentum(self, xs, ys, gamma=0.9, nesterov=False):
        v_prev = np.zeros(self.w.shape)
        for i in tqdm(range(self.n_epochs)):
            for _x, _y in zip(xs, ys):
                _x = np.array([_x])
                _y = np.array([_y])

                if nesterov:
                    dw = self.derivative_nesterov(_x, _y, gamma, v_prev)
                else:
                    dw = self.derivative(_x, _y)
                v_curr = gamma * v_prev + self.learning_rate * dw
                self.w = self.w - v_curr  # don't do -=, leads to bug
                v_prev = v_curr
                self.update_history(xs, ys)

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
        m = p.shape[0]
        dw = np.dot(p, xs ** 2) / m
        return dw

    def derivative_nesterov(self, xs, ys, gamma, v_prev):
        w = self.w - gamma * v_prev
        p = np.dot(xs ** 2, w) - ys
        m = p.shape[0]
        dw = np.dot(p, xs ** 2) / m
        return dw
