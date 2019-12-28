import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class APFBS(object):
    """APFBS
    - w_t = w_{t-1} - eta_t*g_t
    """

    def _fit_Q(self, X, y, coef_, stats_=None):
        opt = Gradient(self.loss)
        Q = 1
        for i in range(X.shape[0]):
            coef_ = coef_ - self.eta0*opt.grad(X[i, :], y[i], coef_, Q)
            if self.penalty == "l1":
                prox = 1 - self.eta0*Q*self.alpha/(np.abs(coef_)+self.eps_)
                coef_ = coef_*prox*(prox > 0)
            Q = 1/(np.abs(coef_)+self.eps_)
            Q = 1/(self.prop + len(coef_)*(1-self.prop)/Q.sum()*Q)
        return [coef_, None]

    def _fit_I(self, X, y, coef_, stats_=None):
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            coef_ = coef_ - self.eta0*opt.grad(X[i, :], y[i], coef_)
            if self.penalty == "l1":
                prox = 1 - self.eta0*self.alpha/(np.abs(coef_)+self.eps_)
                coef_ = coef_*prox*(prox > 0)
        return [coef_, None]


class APFBSClassifier(BaseClassifier, APFBS):

    def __init__(self, eta0=0.1, alpha=10**-4, loss="half-space",
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, prop=None, eps_=10**-8):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)

        self.eps_ = eps_
        if prop is None:
            self._fit = self._fit_I
        else:
            self.prop = prop
            self._fit = self._fit_Q


class APFBSRegressor(BaseRegressor, APFBS):

    def __init__(self, eta0=0.1, power_t=0.5, alpha=10**-4, loss="hyper-plane",
                 penalty=None, fit_intercept=False, warm_start=False,
                 n_jobs=1, prop=None, eps_=10**-8):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)

        self.eps_ = eps_
        if prop is None:
            self._fit = self._fit_I
        else:
            self.prop = prop
            self._fit = self._fit_Q
