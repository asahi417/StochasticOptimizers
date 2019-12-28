import numpy as np

from scipy.sparse import csr_matrix
from .utility import Gradient, Base, BaseRegression


class SDAClassifier(Base):
    """Stochastic Dual Averaging."""

    def __init__(self, loss="log", eta0=0.1, power_t=0, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)

    def _fit_bin(self, X, y, coef_, stats_=None):
        if stats_ is None:
            sum_grad = csr_matrix((1, X.shape[1]), dtype=np.float)
            ind = 0
        else:
            sum_grad = stats_[0]
            ind = stats_[1]
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            sum_grad = sum_grad.tolil()+opt.grad(X[i, :], y[i], coef_).tolil()
            coef_ = -self.eta0/pow(i+1+ind, self.power_t)*sum_grad
            if self.penalty == "l1":
                prox = sum_grad.copy()
                prox[0, [prox.indices]] = \
                    1/(i+1+ind) - self.alpha/np.abs(prox.data)
                prox = prox.multiply(prox > 0)
                coef_ = coef_.multiply(prox)
        return [coef_, [sum_grad, i]]  # coef and sum of gradient


class SDARegressor(BaseRegression):
    """Stochastic Dual Averaging."""

    def __init__(self, loss="square", eta0=0.1, power_t=0.5, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)

    def _fit(self, X, y, coef_, stats_=None):
        _eps = 10**-6
        if stats_ is None:
            sum_grad = np.zeros(X.shape[1])
            ind = 0
        else:
            sum_grad = stats_[0]
            ind = stats_[1]
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            sum_grad += opt.grad(X[i, :], y[i], coef_)
            if self.penalty == "l1":
                prox = 1/(i+1) - self.alpha/(np.abs(sum_grad)+_eps)
                coef_ = - self.eta0 * pow(i+1+ind, self.power_t) * sum_grad \
                                    * prox*(prox > 0)
            else:
                coef_ = -self.eta0/pow(i+1+ind, self.power_t)*sum_grad
        return [coef_, [sum_grad, i+1]]  # coef and sum of gradient
