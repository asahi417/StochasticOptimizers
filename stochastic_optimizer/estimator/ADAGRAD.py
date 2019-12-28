
import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class AdaGrad(object):
    """Adagrad

     Parameters
    --------------------
    type: ["rda", "fbs"]
    """

    def _fit_rda(self, X, y, coef_, stats_=None):
        if stats_ is None:
            sum_grad = np.zeros(coef_.shape)
            g_var = np.zeros(coef_.shape)
            ind = 0
        else:
            sum_grad = stats_[0]
            g_var = stats_[1]
            ind = stats_[2]
        opt = Gradient(self.loss)
        for i in range(len(X)):
            gd = opt.grad(X[i, :], y[i], coef_)
            sum_grad += gd
            g_var += gd*gd
            coef_ = -self.eta0 * sum_grad/(np.sqrt(g_var)+self.eps_)
            if self.penalty == "l1":
                prox = 1 - self.alpha*(i+1+ind)/(np.abs(sum_grad)+self.eps_)
                coef_ = coef_ * prox * (prox > 0)
        return [coef_, [sum_grad, g_var, i+1+ind]]  # coef and sum of gradient

    def _fit_fbs(self, X, y, coef_, stats_=None):
        g_var = np.zeros(coef_.shape) if stats_ is None else stats_
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            g_var += gd*gd

            # update
            if self.penalty == "l1":
                v = np.sqrt(g_var)
                coef_ += -self.eta0*gd/(v+self.eps_)
                prox = 1-self.eta0*self.alpha/(np.abs(coef_)*v+self.eps_)
                coef_ = coef_*prox*(prox > 0)
            else:
                coef_ += -self.eta0*gd/(np.sqrt(g_var)+self.eps_)
        return [coef_, g_var]


class AdaGradRegressor(BaseRegressor, AdaGrad):
    def __init__(self, loss="square", eta0=0.3, alpha=10**-4, n_jobs=1,
                 penalty=None, fit_intercept=True, warm_start=False,
                 eps_=10**-6, solver="fbs"):

        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)

        self.eps_ = eps_
        self.solver = solver
        if self.solver is "rda":
            self._fit = self._fit_rda
        elif self.solver is "fbs":
            self._fit = self._fit_fbs
        else:
            raise ValueError("solver %s is unknown." % solver)


class AdaGradClassifier(BaseClassifier, AdaGrad):
    def __init__(self, loss="square", eta0=0.3, alpha=10**-4, n_jobs=1,
                 penalty=None, fit_intercept=True, warm_start=False,
                 eps_=10**-6, solver="fbs"):

        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)

        self.eps_ = eps_
        self.solver = solver
        if self.solver == "rda":
            self._fit = self._fit_rda
        elif self.solver == "fbs":
            self._fit = self._fit_fbs
        else:
            raise ValueError("solver %s is unknown." % solver)
