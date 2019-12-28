import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class SGD(object):
    """StochasticGradientDescent

    - w_t = w_{t-1} - eta_t*g_t
    """

    def _fit(self, X, y, coef_, stats_=None):
        ind = stats_ if stats_ is not None else 0
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            if self.power_t == 0:
                coef_ += - self.eta0*opt.grad(X[i, :], y[i], coef_)
            else:
                coef_ += - self.eta0/pow(i+1+ind, self.power_t) \
                                        * opt.grad(X[i, :], y[i], coef_)
            if self.penalty == "l1":
                if self.power_t == 0:
                    prox = 1 - self.eta0*self.alpha/(np.abs(coef_)+self.eps_)
                else:
                    prox = 1 - self.eta0*self.alpha \
                        / (np.abs(coef_*pow(i+1+ind, self.power_t))+self.eps_)
                coef_ = coef_ * prox * (prox > 0)
        return [coef_, i+1+ind]


class SGDClassifier(BaseClassifier, SGD):

    def __init__(self, loss="log", eta0=0.1, power_t=0, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, eps_=10**-6):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)
        self.eps_ = eps_


class SGDRegressor(BaseRegressor, SGD):

    def __init__(self, loss="square", eta0=0.1, power_t=0.5, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, eps_=10**-6):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)
        self.eps_ = eps_
