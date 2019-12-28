import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class SDA(object):
    """Stochastic Dual Averaging
    - g^^ += g_t
    - w_t = w_{t-1} - eta_t*g^^/n
    """

    def _fit(self, X, y, coef_, stats_=None):
        if stats_ is None:
            sum_grad = np.zeros(X.shape[1])
            ind = 0
        else:
            sum_grad = stats_[0]
            ind = stats_[1]

        _eps = 10**-6
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            sum_grad += opt.grad(X[i, :], y[i], coef_)
            if self.penalty == "l1":
                prox = 1/(i+1) - self.alpha/(np.abs(sum_grad)+_eps)
                coef_ = - self.eta0 * pow(i+1+ind, self.power_t) * sum_grad \
                                    * prox*(prox > 0)
            else:
                coef_ = -self.eta0/pow(i+1+ind, self.power_t)*sum_grad
        return [coef_, [sum_grad, i]]


class SDAClassifier(BaseClassifier, SDA):
    def __init__(self, loss="log", eta0=0.1, power_t=0, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)


class SDARegressor(BaseRegressor, SDA):
    def __init__(self, loss="square", eta0=0.1, power_t=0.5, alpha=10**-4,
                 penalty=None, fit_intercept=False, warm_start=False,
                 n_jobs=1):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)
