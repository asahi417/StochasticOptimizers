import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class RMSprop(object):
    """RMSprop

    - w_t = w_{t-1} - eta_t*g_t
    - eta_t = 1 / RMS(g_t)
    - RMS(g_t) = (E[g^2]_t)^{1/2}
    """

    def _fit(self, X, y, coef_, stats_=None):
        if stats_ is None:
            ini = True
        else:
            ini = False
            g_var = stats_[0]
            g_mom = stats_[1]
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            if ini:
                # initialization
                ini = False
                g_mom = gd.copy()
                g_var = gd*gd
            else:
                # momentum
                g_mom = self.momentum*gd + (1-self.momentum)*g_mom
                # mean squared gradient and its bias
                g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
            # update
            # if self.penalty == "l1":
            #     v = np.sqrt(g_var)+self.eps_
            #     coef_ += -self.eta0*g_mom/(v)
            #     prox = 1-self.eta0*self.alpha/(np.abs(coef_)*v)
            #     coef_ = coef_*prox*(prox > 0)
            # else:
            coef_ += -self.eta0*g_mom/(np.sqrt(g_var)+self.eps_)
        return [coef_, [g_var, g_mom]]


class RMSpropRegressor(BaseRegressor, RMSprop):
    def __init__(self, loss="square", eta0=0.3, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, momentum=1, var_w=0.1, eps_=10**-6):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)
        self.momentum = momentum
        self.var_w = var_w
        self.eps_ = eps_


class RMSpropClassifier(BaseClassifier, RMSprop):
    def __init__(self, loss="log", eta0=0.3, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, momentum=1, var_w=0.1, eps_=10**-6):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)
        self.momentum = momentum
        self.var_w = var_w
        self.eps_ = eps_
