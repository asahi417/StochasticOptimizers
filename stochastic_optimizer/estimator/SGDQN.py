import numpy as np

from .utility import Gradient, Base, BaseRegression


class SGDQN(object):
    """ Underconstruction
    AdaDelta, VSGD みたいに, rms, var_w で場合分けして拡張
    """

    def _fit(self, X, y, coef_, stats_=None):
        _eps = 10**-6
        if stats_ is None:
            ini = True
        else:
            ini = False
            ind = stats_[0]
            h = stats_[1]
            gd_ = stats_[2]
            var_w = stats_[3]
        opt = Gradient(self.loss)
        coef__ = coef_.copy()
        eps = 10**-8
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            if ini:
                # initialization
                ini = False
                ind = 0
                var_w = 2
                h = self.eta0
            else:
                coef_ += - gd*h/pow(i+1+ind, self.power_t)
                # Finite Differenced Hessian (momentum average)
                if self.decay:
                    if np.mod(i+1+ind, self.skip) == 0:
                        h = 2/var_w*(coef_-coef__)/(gd-gd_+eps) + (1-2/var_w)*h
                        h[h < self.eta0] = self.eta0
                        var_w += 1
                else:
                    h = np.abs((coef_-coef__)/(gd-gd_+eps))

            # regularization
            # if self.penalty == "l1":
            #     prox = 1 - self.eta0 * \
            #         self.alpha/(np.abs(coef_*pow(i+1+ind, self.power_t))+eps)
            #     coef_ = coef_ * prox * (prox > 0)
            gd_, coef__ = gd.copy(), coef_.copy()
        return [coef_, [i+1, h, gd_, var_w]]


class SGDQNClassifier(Base, SGDQN):
    def __init__(self, loss="log", eta0=0.1, power_t=1, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, skip=10, decay=False):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)
        self.skip = skip
        self.decay = decay


class SGDQNRegressor(BaseRegression, SGDQN):
    def __init__(self, loss="square", eta0=0.1, power_t=1, alpha=10**-4,
                 penalty=None, fit_intercept=False, warm_start=False,
                 n_jobs=1, skip=10, decay=False):
        self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)
        self.skip = skip
        self.decay = decay
