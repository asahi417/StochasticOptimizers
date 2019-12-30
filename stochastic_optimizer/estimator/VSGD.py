import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class VSGD(object):
    """VSGD (variance SGD)

    - w_t = w_{t-1} - eta_t*g_t
    - eta_t = h_t/h^2_t * (g_t)^2/g^2_t
    - Instantaneous Hessian "h_t" is approximation by FDM
    - Reflect variance of gradient and approx hessian

     Parameters
    ----------
    var_w: weight of expectation
    rms: for g_var(variance of gradient) and coef_diff(difference of coef)
        True
            g_t = var_w*new_gradient + (1-var_w)*g_t
            g^2_t = var_w*(new_gradient)^2 + (1-var_w)*g^2_t
            h_t = var_w*new_hessian + (1-var_w)*h_t
            h^2_t = var_w*(new_hessian)^2 + (1-var_w)*h^2_t
            -> learning rate = h_t/h^2t * (g_t)^2/g^2t

        False
            g_t += new_gradient
            g^2_t += (new_gradient)^2
            h_t += new_hessian
            h^2_t += (new_hessian)^2
            -> learning rate = h_t/h^2t * (g_t)^2/g^2t

        if "rms" is True and "var_w" == 1
            g_t = g_t, g^2_t = (new_gradient)^2
            h_t = new_hessian, h^2_t = (new_hessian)^2
            -> learning rate = 1/h_t [Simplest and Effective !!]

    decay: original algorithm employs heuristic decaying method for var_w
        True -> var_w = (1-g^2/g_var)var_w + 1
        False -> var_w = var_w

     Return
    ----------
    coef_: coefficient, ndarray, shape(class_n, order)
    g_mom: momentum gradient
    g_ave: g_t
    g_var: g^2_t
    h_ave: h_t
    h_var: h^2_t

     Comment
    ----------
    * regularization is not coded

     Bugs
    ----------
    * only work wheb rms=True and var_w=1
    """

    def _fit(self, X, y, coef_, stats_=None):
        if stats_ is None:
            ini = True
        else:
            ini = False
            g_mom = stats_[0]
            # not used in case of (rms is True and var_w == 1)
            g_ave = stats_[1]
            g_var = stats_[2]
            h_ave = stats_[3]
            h_var = stats_[4]

        # gradient generator instance
        opt = Gradient(self.loss)

        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            if ini:
                # Initialization
                ini = False
                g_mom = gd.copy()
                # -not used in case of (rms is True and var_w == 1)
                g_ave = gd.copy()
                g_var = gd*gd
                h_ave = np.abs(
                    (gd-opt.grad(X[i, :], y[i], coef_+gd))/(gd+self.eps_))
                h_var = h_ave*h_ave
            else:
                # momentum
                g_mom = self.momentum*gd + (1-self.momentum)*g_mom

                if self.rms and self.var_w == 1:
                    # Instantaneous Hessian approximation (FDM)
                    h_ = np.abs(
                        (gd-opt.grad(X[i, :], y[i], coef_+gd))/(gd+self.eps_))
                    lr = 1/(h_+self.eps_)
                    # Update
                    coef_ += -lr*g_mom
                else:
                    # Instantaneous Hessian approximation (FDM)
                    h_ = np.abs(
                        (gd-opt.grad(X[i, :], y[i], coef_+g_ave))/(g_ave+self.eps_))
                    if self.rms:
                        # Variance bias of gradient
                        g_ave = gd/(self.var_w+self.eps_) \
                            + g_ave*(1-1/(self.var_w+self.eps_))
                        g_var = gd*gd/(self.var_w+self.eps_) \
                            + g_var*(1-1/(self.var_w+self.eps_))
                        var_bias = g_ave*g_ave/(g_var+self.eps_)

                        # Variance bias of hessian
                        h_ave = h_/(self.var_w+self.eps_) \
                            + h_ave*(1-1/(self.var_w+self.eps_))
                        h_var = h_*h_/(self.var_w+self.eps_) \
                            + h_var*(1-1/(self.var_w+self.eps_))

                        lr = h_ave/(h_var+self.eps_)*var_bias
                        # Update
                        coef_ += -lr*g_mom

                        # Weight update
                        if self.decay:
                            self.var_w = (1-var_bias)*self.var_w + 1
                    else:
                        # Variance bias of gradient
                        g_ave += gd
                        g_var += gd*gd

                        # Variance bias of hessian
                        h_ave += h_
                        h_var += h_*h_
                        lr = h_ave*(g_ave*g_ave)/(h_var*g_var+self.eps_)
                        # Update
                        coef_ += -lr*g_mom
                # if self.penalty == "l1":
                #     prox = 1-lr*g_mom*self.alpha/(
                #              np.abs(coef_)+self.eps_)
                #     coef_ = coef_*prox*(prox > 0)
        return [coef_, [g_mom, g_ave, g_var, h_ave, h_var]]


class VSGDRegressor(BaseRegressor, VSGD):
    def __init__(self, loss="square", eps_=10**-6,
                 fit_intercept=True, warm_start=False,
                 n_jobs=1, momentum=1, var_w=1, decay=False, rms=True, **kwargs):
        self.set_params(loss=loss,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)
        self.eps_ = eps_
        self.momentum = momentum
        self.rms = rms
        self.var_w = var_w
        self.decay = decay


class VSGDClassifier(BaseClassifier, VSGD):
    def __init__(self, loss="log", eps_=10**-6,
                 fit_intercept=True, warm_start=False,
                 n_jobs=1, momentum=1, var_w=1, decay=False, rms=True, **kwargs):
        self.set_params(loss=loss,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)
        self.eps_ = eps_
        self.momentum = momentum
        self.rms = rms
        self.var_w = var_w
        self.decay = decay
