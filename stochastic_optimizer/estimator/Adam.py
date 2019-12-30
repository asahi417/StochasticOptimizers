import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class Adam(object):
    """Adam

    - w_t = w_{t-1} - eta_t*g_t
    - eta_t = moment bias / RMS(g_t)
    - RMS(g_t) = (E[g^2]_t)^{1/2}

     Parameters
    ----------
    var_w: weight of expectation
    rms: for g_var(variance of gradient) and coef_diff(difference of coef)
        True
            E[g^2_t] ~ g^2_t = var_w*new_gradient^2 + (1-var_w)*g^2_t

        False
            E[g^2_t] ~ g^2_t += new_gradient^2

        if "rms" is True and "var_w" == 1
            E[g^2_t] ~ g^2_t = new_gradient^2

     Return
    ----------
    coef_: coefficient, ndarray, shape(class_n, order)
    g_mom: momentum gradient
    g_var: E[g^2_t]
    coef_diff: E[delta_{t-1}^2]


     Comment
    ----------
    * regularization is not coded
    """

    def _fit(self, X, y, coef_, stats_=None):
        if stats_ is None:
            ini = True
        else:
            ini = False
            ind = stats_[0]
            g_var = stats_[1]
            g_mom = stats_[2]
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            if ini:
                # initialization
                ini = False
                ind = 0
                g_mom = gd.copy()
                g_var = gd*gd
                bias_a = 1
                bias_v = 1
            else:
                # momentum and its bias
                g_mom = self.momentum*gd + (1-self.momentum)*g_mom
                bias_a = 1-(1-self.momentum)**(ind+1+i)
                # mean squared gradient and its bias
                g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
                bias_v = 1-(1-self.var_w)**(ind+1+i)
            # update
            lr = self.eta0*bias_a/(np.sqrt(g_var*bias_v)+self.eps_)
            coef_ += -lr*g_mom
            if self.penalty == "l1":
                prox = 1-lr*g_mom*self.alpha/(
                         np.abs(coef_)+self.eps_)
                coef_ = coef_*prox*(prox > 0)
        return [coef_, [i+1+ind, g_var, g_mom]]


class AdamRegressor(BaseRegressor, Adam):
    def __init__(self, loss="square", eta0=0.3, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, momentum=1, var_w=0.1, eps_=10**-8):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)
        self.momentum = momentum
        self.var_w = var_w
        self.eps_ = eps_


class AdamClassifier(BaseClassifier, Adam):
    def __init__(self, loss="log", eta0=0.3, alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, momentum=1, var_w=0.1, eps_=10**-8):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)
        self.momentum = momentum
        self.var_w = var_w
        self.eps_ = eps_
