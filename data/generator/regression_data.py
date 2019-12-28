import numpy as np


class RegressionData(object):
    """Generate data for regression.

     Parameter
    ----------------
    shape: (size, order)
    data heterogeneity parameters:
        rate: ratio of the data which are not frequently 0
              稀にしか現れない要素の割合
        prob: probability of the 0 data
              稀にしか現れない要素の0である確率
        dist: distribution of input data
              None (uniform), Normal

    coef scale parameters:
        c_var: variance of distribution
        c_mean: mean of distribution
        c_rate: ratio of zero elements
        c_dist: distribution of input data
                None (uniform), Normal

    noise_var: noise for output data

     Attribution
    ---------------
    X: input
    X_var: variance
    X_mean: mean
    X_sparsity: ratio of 0 elements
    X_density: ratio of nonzero elements
               size 方向に見た時に一度も0にならない成分の割合
    X_densevar: size 方向に見た時に一度も0にならない成分の variance

    coef_: coefficient

    y: output
    """

    def __init__(self, var=1, mean=4, rate=0.7, prob=0.5, dist="normal",
                 shape=(100, 10), c_var=1, c_mean=0.1, c_rate=0.2,
                 c_dist=None, noise_var=None):
        size, order = shape
        self.X = self.generator_X(var, mean, rate, prob, dist, size, order)
        self.coef_ = self.generator_coef(c_var, c_mean, c_rate, c_dist, order)
        self.y = self.X.dot(self.coef_.reshape(-1, 1))[:, 0]
        if noise_var is not None:
            self.y_noise = self.y+np.random.normal(0, np.sqrt(noise_var), size)

    def generator_X(self, var, mean, rate, prob, dist, size, order):
        if dist in (None, "uniform"):
            X = np.random.rand(size, order)-0.5
            X = X*var+mean
        elif dist == "normal":
            X = np.random.normal(mean, np.sqrt(var), (size, order))
        r = np.int(np.ceil(order*rate))
        p = np.int(np.ceil(r*prob))
        for i in range(size):
            X[i, np.sort([np.random.choice([i for i in range(r)],
                                           p, replace=False)])] = np.zeros(p)
        self.X_var = X.var()
        self.X_mean = X.mean()
        self.X_sparsity = p/order
        self.X_density = ((X == 0).sum(0) == 0).sum()/order
        self.X_densevar = (X[X != 0]).var()
        return X

    def generator_coef(self, c_var, c_mean, c_rate, c_dist, order):
        if c_dist in (None, "uniform"):
            coef = (np.random.rand(order)-0.5+c_mean)*c_var
        elif c_dist == "normal":
            coef = np.random.normal(c_mean, np.sqrt(c_var), order)
        r = np.int(np.ceil(order*c_rate))
        coef[np.sort([np.random.choice([i for i in range(order)], r,
                                       replace=False)])] = np.zeros(r)
        self.coef_var = coef.var()
        self.coef_mean = coef.mean()
        self.coef_sparsity = r/order
        self.coef_densevar = (coef[coef != 0]).var()
        return coef
