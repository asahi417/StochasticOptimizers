import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class AdaDelta(object):
    """AdaDelta

    - w_t = w_{t-1} - eta_t*g_t
    - eta_t = RMS(g_t)/RMS(delta_{t-1})
    - delta_{t-1} = w_{t-1} - w_{t-2}
                  = eta_{t-1}*g_{t-1}
    - RMS(g_t) = (E[g^2]_t)^{1/2}

     Parameters
    ----------
    eps_: numerical stability and initial learning rate
        eta_0 = (eps_/(g^2))^{1/2}
    var_w: weight of expectation
    rms: for g_var(variance of gradient) and coef_diff(difference of coef)
        True
            E[g^2_t] ~ g^2_t = var_w*new_gradient^2 + (1-var_w)*g^2_t
            E[delta_{t-1}^2] ~ delta_{t-1}^2 = var_w*(eta_{t-1}*g_{t-1})^2
                                                 + (1-var_w)*delta_{t-2}^2

        False
            E[g^2_t] ~ g^2_t += new_gradient^2
            E[delta_{t-1}^2] ~ delta_{t-1}^2 += (eta_{t-1}*g_{t-1})^2

        if "rms" is True and "var_w" == 1
            E[g^2_t] ~ g^2_t = new_gradient^2
            E[delta_{t-1}^2] ~ delta_{t-1}^2 = (eta_{t-1}*g_{t-1})^2
            -> eta_t = g_t/eta_{t-1}*g_{t-1}

     Return
    ----------
    coef_: coefficient, ndarray, shape(class_n, order)
    g_mom: momentum gradient
    g_var: E[g^2_t]
    coef_diff: E[delta_{t-1}^2]


     Comment
    ----------
    * sensitive for eps_ (like learning rate in other algorithm)
    * regularization is not coded
    """

    def _fit(self, X, y, coef_, stats_=None):
        if stats_ is None:
            ini = True
        else:
            ini = False
            g_mom = stats_[0]
            g_var = stats_[1]
            coef_diff = stats_[2]

        # gradient generator instance
        opt = Gradient(self.loss)

        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            if ini:
                # Initialization
                ini = False
                g_mom = gd.copy()
                g_var = gd*gd
                h = np.sqrt((self.eps_)/(g_var+self.eps_))
                update = -g_mom*h
                coef_diff = update**2
            else:
                # momentum
                g_mom = self.momentum*gd + (1-self.momentum)*g_mom

                if self.rms and self.var_w == 1:
                    # Instantaneous Hessian approximation (FDM)
                    h = (coef_diff+self.eps_)/(gd+self.eps_)
                    # Difference of coefficient
                    coef_diff = g_mom*h
                elif self.rms:
                    # Variance of gradient
                    g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
                    # Instantaneous Hessian approximation (FDM)
                    h = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
                    # Difference of coefficient
                    coef_diff = self.var_w*((g_mom*h)**2) \
                        + (1-self.var_w)*coef_diff
                else:
                    # Variance of gradient
                    g_var += gd*gd
                    # Instantaneous Hessian approximation (FDM)
                    h = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
                    # Difference of coefficient
                    coef_diff += (g_mom*h)**2

            # Update
            coef_ += -g_mom*h
            if self.penalty == "l1":
                prox = 1-h*self.alpha/(np.abs(coef_)+self.eps_)
                coef_ = coef_*prox*(prox > 0)
        return [coef_, [g_mom, g_var, coef_diff]]


class AdaDeltaRegressor(BaseRegressor, AdaDelta):
    def __init__(self, loss="square", alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, momentum=1, var_w=0.1, eps_=10**-8, rms=True):
        self.set_params(loss=loss, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)
        self.momentum = momentum
        self.var_w = var_w
        self.eps_ = eps_
        self.rms = rms


class AdaDeltaClassifier(BaseClassifier, AdaDelta):
    def __init__(self, loss="log", alpha=10**-4,
                 penalty=None, fit_intercept=True, warm_start=False,
                 n_jobs=1, momentum=1, var_w=0.1, eps_=10**-8, rms=True):
        self.set_params(loss=loss, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)
        self.momentum = momentum
        self.var_w = var_w
        self.eps_ = eps_
        self.rms = rms
