import numpy as np


class Gradient(object):
    """Generate gradient.

     Parameters
    ----------
    x : input (csr_matrix, 1 x n) or (ndarray, n)
    y : output (scalar, 1)
    weight : estimated weight (csr_matrix, 1 x n) or (ndarray, n)

     Return
    ----------
    grad : gradient (csr_matrix, 1 x n) """

    def __init__(self, loss):
        if loss == "log":
            self.grad = self.__grad_log
        elif loss == "hinge":
            self.grad = self.__grad_hinge
        elif loss == "half-space":
            self.grad = self.__grad_halfspace
        elif loss == "square":
            self.grad = self.__grad_square
        elif loss == "hyper-plane":
            self.grad = self.__grad_hyperplane

    def __grad_square(self, x, y, weight):
        return (x.dot(weight)-y)*x

    def __grad_hyperplane(self, x, y, weight, metric=None):
        if metric is None:
            return (x.dot(weight)-y)*x/x.dot(x)
        else:
            x_ = x*metric
            return (x.dot(weight)-y)*x_/x.dot(x_)

    def __grad_log(self, x, y, weight):
        s = y*x.dot(weight)
        return np.zeros(x.shape) if s > 100 else -x*y/(1+np.exp(s))

    def __grad_hinge(self, x, y, weight, th=1):  # th=0 -> perceptron
        return -x*y if y*x.dot(weight) < th else np.zeros(x.shape)

    def __grad_halfspace(self, x, y, weight, metric=None):
        if metric is None:
            error = 1 - y*x.dot(weight)
            return -x*error/y/x.dot(x) if error > 0 else np.zeros(x.shape)
        else:
            error = 1 - y*x.dot(weight)
            return -x*error/y/x.dot(x) if error > 0 else np.zeros(x.shape)


if __name__ == '__main__':
    v = np.array([0.1, 0, 0.1])
    g = Optimizer(loss="half-space")
    gr = g.grad(v, 4, v)
    print(gr)
