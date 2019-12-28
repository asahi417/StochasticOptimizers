import numpy as np

from scipy.sparse import csr_matrix

from .utility import Gradient
from .SGD import SGDClassifier, SGDRegressor


class APFBSClassifier(SGDClassifier):
    """simple Adaptive Forward Backword Splitting
    (Iterative Projection onto Half-space)."""

    def __init__(self, eta0=0.1, alpha=10**-4,
                 loss=None, power_t=None,
                 penalty='l1', fit_intercept=True, warm_start=False):
        super().__init__(loss="half-space", eta0=eta0, power_t=0,
                         penalty=penalty, fit_intercept=fit_intercept,
                         warm_start=warm_start, alpha=alpha)


class APFBSRegressor(SGDRegressor):
    """simple Adaptive Forward Backword Splitting
    (Iterative Projection onto Half-space)."""

    def __init__(self, eta0=0.1, alpha=10**-4,
                 penalty='l1', fit_intercept=True, warm_start=False,
                 loss=None, power_t=None, n_jobs=1, prop=None):

        self.set_params(loss="hyper-plane", eta0=eta0, power_t=0, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)
        self.prop = prop

    def _fit(self, X, y, coef_):
        opt = Gradient(self.loss)
        Q = 1
        for i in range(X.shape[0]):
            if self.prop is not None:
                Q = 1/(np.abs(coef_)+10**-6)
                Q = 1/(self.prop + len(coef_)*(1-self.prop)/Q.sum()*Q)
            coef_ += - self.eta0 * opt.grad(X[i, :], y[i], coef_, Q)
            if self.penalty == "l1":
                prox = 1 - self.eta0 * self.alpha * Q / (np.abs(coef_)+10**-6)
                coef_ = coef_ * prox * (prox > 0)
        return [coef_]
