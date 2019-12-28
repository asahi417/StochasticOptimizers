import numpy as np
from .utility import Gradient, BaseClassifier, BaseRegressor


class PDA(object):
    """Projection-based Dual Averaging
    - g^^ += g_t
    - w_t = w_{t-1} - eta_t*g^^/n

    _fit_D: AdaDelta rule for learning rate
    _fit_Q: proportionate metric
    _fit_DQ: AdaDelta learning rate and proportionate metric
    _fit_I: vanilla version
    _fit_R: RDA based version (coefficient of regularization
            increase depend on time)
    """

    def _fit_D2(self, X, y, coef_, stats_=None):
        """AdaDelta metric"""
        if stats_ is None:
            sum_grad = np.zeros(X.shape[1])
            g_var = np.zeros(X.shape[1])
            coef_diff = np.zeros(X.shape[1])
            ini = True
        else:
            sum_grad = stats_[0]
            g_var = stats_[1]
            coef_diff = stats_[2]
            ini = False

        opt = Gradient(self.loss)
        h_ = 1
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_, h_)
            # AdaDelta learning rate
            if ini:
                g_var += gd*gd
                coef_diff += (gd*h_)**2
                ini = False
            else:
                if self.var_w is None:
                    # Variance of gradient
                    g_var += gd*gd
                    # Instantaneous Hessian approximation (FDM)
                    h_ = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
                    # Difference of coefficient
                    coef_diff += (gd*h_)**2
                else:
                    # Variance of gradient
                    g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
                    # Instantaneous Hessian approximation (FDM)
                    h_ = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
                    # Difference of coefficient
                    coef_diff = \
                        self.var_w*((gd*h_)**2) + (1-self.var_w)*coef_diff
                h_ = len(h_)*h_/(h_.sum()+self.eps_)
            # UPDATE and PROX
            sum_grad += gd
            if self.penalty == "l1":
                prox = 1 - self.alpha/(np.abs(sum_grad)+self.eps_)
                coef_ = - self.eta0*sum_grad*prox*(prox > 0)
            else:
                coef_ = -self.eta0*sum_grad
        return [coef_, [sum_grad, g_var, coef_diff]]

    def _fit_A(self, X, y, coef_, stats_=None):
        """Adam learning rate"""
        if stats_ is None:
            sum_grad = np.zeros(X.shape[1])
            g_var = np.zeros(X.shape[1])
            ind = 0
        else:
            sum_grad = stats_[0]
            g_var = stats_[1]
            ind = stats_[2]

        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            if ind == 0 or self.var_w == 1:
                # Variance of gradient
                g_var += gd*gd
                bias_v = 1
            else:
                # Variance of gradient
                g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
                bias_v = 1-(1-self.var_w)**(ind+1+i)
            lr = self.eta0/(np.sqrt(g_var*bias_v)+self.eps_)
            # UPDATE and PROX
            sum_grad += gd
            if self.penalty == "l1":
                prox = 1 - self.alpha/(np.abs(sum_grad)+self.eps_)
                coef_ = - lr*sum_grad*prox*(prox > 0)
            else:
                coef_ = -lr*sum_grad
        return [coef_, [sum_grad, g_var, ind+1+i]]

    def _fit_V(self, X, y, coef_, stats_=None):
        """VSGD learning rate"""
        if stats_ is None:
            sum_grad = np.zeros(X.shape[1])
            g_ave, g_var = np.zeros(X.shape[1]), np.zeros(X.shape[1])
            h_ave, h_var = np.zeros(X.shape[1]), np.zeros(X.shape[1])

        else:
            sum_grad = stats_[0]
            g_ave = stats_[1]
            g_var = stats_[2]
            h_ave = stats_[3]
            h_var = stats_[4]

        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            # AdaDelta learning rate
            if self.var_w == 1:
                # Instantaneous Hessian approximation (FDM)
                lr = np.abs(gd/(gd-opt.grad(X[i, :], y[i],
                                            coef_+gd)+self.eps_))
            elif self.var_w is None:
                # Variance of gradient
                h = np.abs((gd-opt.grad(X[i, :], y[i], coef_+gd)) /
                           (gd+self.eps_))
                g_var += gd*gd
                g_ave += gd
                g_bias = g_ave**2/(g_var+self.eps_)

                h_var += h*h
                h_ave += h
                h_bias = g_ave/(g_var+self.eps_)

                lr = h_bias*g_bias
            else:
                # Variance of gradient
                h = np.abs((gd-opt.grad(X[i, :], y[i], coef_+gd)) /
                           (gd+self.eps_))
                g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
                g_ave = self.var_w*gd + (1-self.var_w)*g_ave
                g_bias = g_ave**2/(g_var+self.eps_)

                h_var = self.var_w*h*h + (1-self.var_w)*h_var
                h_ave = self.var_w*h + (1-self.var_w)*h_ave
                h_bias = g_ave/(g_var+self.eps_)

                lr = h_bias*g_bias
            # UPDATE and PROX
            sum_grad += gd
            if self.penalty == "l1":
                prox = 1 - self.alpha/(np.abs(sum_grad)+self.eps_)
                coef_ = - lr*sum_grad*prox*(prox > 0)
            else:
                coef_ = -lr*sum_grad
        return [coef_, [sum_grad, g_ave, g_var, h_ave, h_var]]

    def _fit_D(self, X, y, coef_, stats_=None):
        """AdaDelta learning rate"""
        if stats_ is None:
            sum_grad = np.zeros(X.shape[1])
            g_var = np.zeros(X.shape[1])
            coef_diff = np.zeros(X.shape[1])
            ini = True
        else:
            sum_grad = stats_[0]
            g_var = stats_[1]
            coef_diff = stats_[2]
            ini = False

        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            # AdaDelta learning rate
            if ini:
                g_var += gd*gd
                if self.eta0 is not None:
                    h = self.eta0
                else:
                    h = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
                coef_diff += (gd*h)**2
                ini = False
            else:
                if self.var_w is None or ini:
                    # Variance of gradient
                    g_var += gd*gd
                    # Instantaneous Hessian approximation (FDM)
                    h = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
                    # Difference of coefficient
                    coef_diff += (gd*h)**2
                else:
                    # Variance of gradient
                    g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
                    # Instantaneous Hessian approximation (FDM)
                    h = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
                    # Difference of coefficient
                    coef_diff = \
                        self.var_w*((gd*h)**2) + (1-self.var_w)*coef_diff
            # UPDATE and PROX
            sum_grad += gd
            if self.penalty == "l1":
                prox = 1 - self.alpha/(np.abs(sum_grad)+self.eps_)
                coef_ = - h*sum_grad*prox*(prox > 0)
            else:
                coef_ = -h*sum_grad
        return [coef_, [sum_grad, g_var, coef_diff]]

    def _fit_Q(self, X, y, coef_, stats_=None):
        """proportionate metric"""
        sum_grad = np.zeros(X.shape[1]) if stats_ is None else stats_
        opt = Gradient(self.loss)
        Q = 1
        for i in range(X.shape[0]):
            sum_grad += opt.grad(X[i, :], y[i], coef_, Q)
            if self.penalty == "l1":
                prox = 1 - self.alpha*Q/(np.abs(sum_grad)+self.eps_)
                coef_ = - self.eta0*sum_grad*prox*(prox > 0)
            else:
                coef_ = -self.eta0*sum_grad
            Q = 1/(np.abs(coef_)+self.eps_)
            Q = 1/(self.prop + len(coef_)*(1-self.prop)/Q.sum()*Q)
        return [coef_, sum_grad]

    def _fit_I(self, X, y, coef_, stats_=None):
        """identity metric"""
        sum_grad = np.zeros(X.shape[1]) if stats_ is None else stats_
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            sum_grad += opt.grad(X[i, :], y[i], coef_)
            if self.penalty == "l1":
                prox = 1-self.alpha/(np.abs(sum_grad)+self.eps_)
                coef_ = - self.eta0*sum_grad*prox*(prox > 0)
            else:
                coef_ = -self.eta0*sum_grad
        return [coef_, sum_grad]


class PDAClassifier(BaseClassifier, PDA):
    def __init__(self, eta0=None, alpha=10**-4, penalty=None,
                 fit_intercept=True, warm_start=False, prop=None,
                 eps_=10**-6, loss="half-space", n_jobs=1, var_w=None,
                 solver=None):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)

        self.eps_ = eps_
        self.solver = solver
        if self.solver is None:
            self._fit = self._fit_I
        elif self.solver == "AdaDelta2":
            self.var_w = var_w
            self._fit = self._fit_D2
        elif self.solver == "AdaDelta":
            self.var_w = var_w
            self._fit = self._fit_D
        elif self.solver == "Adam":
            self.var_w = var_w
            self._fit = self._fit_A
        elif self.solver == "vsgd":
            self.var_w = var_w
            self._fit = self._fit_V
        elif self.solver == "prop":
            self.prop = prop
            self._fit = self._fit_Q


class PDARegressor(BaseRegressor, PDA):
    def __init__(self, eta0=0.1, alpha=10**-4, n_jobs=1,
                 penalty=None, fit_intercept=True, warm_start=False,
                 prop=None, eps_=10**-6, loss="hyper-plane", var_w=0.1,
                 solver=None):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)

        self.eps_ = eps_

        if solver is None:
            self._fit = self._fit_I
        elif solver == "AdaDelta":
            self.var_w = var_w
            self._fit = self._fit_D
        elif self.solver == "Adam":
            self.var_w = var_w
            self._fit = self._fit_A
        elif solver == "vsgd":
            self.var_w = var_w
            self._fit = self._fit_V
        elif solver == "prop":
            self.prop = prop
            self._fit = self._fit_Q

        # def _fit_VQ(self, X, y, coef_, stats_=None):
        #     """VSGD learning rate"""
        #     if stats_ is None:
        #         sum_grad = np.zeros(X.shape[1])
        #         g_ave, g_var = np.zeros(X.shape[1]), np.zeros(X.shape[1])
        #         h_ave, h_var = np.zeros(X.shape[1]), np.zeros(X.shape[1])
        #     else:
        #         sum_grad = stats_[0]
        #         g_ave = stats_[1]
        #         g_var = stats_[2]
        #         h_ave = stats_[3]
        #         h_var = stats_[4]
        #
        #     opt = Gradient(self.loss)
        #     Q = 1
        #     for i in range(X.shape[0]):
        #         gd = opt.grad(X[i, :], y[i], coef_, Q)
        #         # AdaDelta learning rate
        #         if self.var_w == 1:
        #             # Instantaneous Hessian approximation (FDM)
        #             lr = np.abs(gd/(gd-opt.grad(X[i, :], y[i],
        #                                         coef_+gd)+self.eps_))
        #         elif self.var_w is None:
        #             # Variance of gradient
        #             h = np.abs((gd-opt.grad(X[i, :], y[i], coef_+gd)) /
        #                        (gd+self.eps_))
        #             g_var += gd*gd
        #             g_ave += gd
        #             g_bias = g_ave**2/(g_var+self.eps_)
        #             h_var += h*h
        #             h_ave += h
        #             h_bias = g_ave/(g_var+self.eps_)
        #
        #             lr = h_bias*g_bias
        #         else:
        #             h = np.abs((gd-opt.grad(X[i, :], y[i], coef_+gd)) /
        #                        (gd+self.eps_))
        #             # Variance of gradient
        #             g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
        #             g_ave = self.var_w*gd + (1-self.var_w)*g_ave
        #             g_bias = g_ave**2/(g_var+self.eps_)
        #
        #             h_var = self.var_w*h*h + (1-self.var_w)*h_var
        #             h_ave = self.var_w*h + (1-self.var_w)*h_ave
        #             h_bias = g_ave/(g_var+self.eps_)
        #
        #             lr = h_bias*g_bias
        #         # UPDATE and PROX
        #         sum_grad += gd
        #         if self.penalty == "l1":
        #             prox = 1 - self.alpha*Q/(np.abs(sum_grad)+self.eps_)
        #             coef_ = - lr*sum_grad*prox*(prox > 0)
        #         else:
        #             coef_ = -lr*sum_grad
        #         Q = 1/(np.abs(coef_)+self.eps_)
        #         Q = 1/(self.prop + len(coef_)*(1-self.prop)/Q.sum()*Q)
        #     return [coef_, [sum_grad, g_ave, g_var, h_ave, h_var]]
    # def _fit_DQ(self, X, y, coef_, stats_=None):
    #     """AdaDelta learning rate and prop metric"""
    #     if stats_ is None:
    #         sum_grad = np.zeros(X.shape[1])
    #         g_var = np.zeros(X.shape[1])
    #         coef_diff = np.zeros(X.shape[1])
    #     else:
    #         sum_grad = stats_[0]
    #         g_var = stats_[1]
    #         coef_diff = stats_[2]
    #
    #     opt = Gradient(self.loss)
    #     Q = 1
    #     for i in range(X.shape[0]):
    #         gd = opt.grad(X[i, :], y[i], coef_, Q)
    #         # AdaDelta learning rate
    #         if self.var_w == 1:
    #             # Instantaneous Hessian approximation (FDM)
    #             h = (coef_diff+self.eps_)/(gd+self.eps_)
    #             # Difference of coefficient
    #             coef_diff = gd*h
    #         elif self.var_w is None:
    #             # Variance of gradient
    #             g_var += gd*gd
    #             # Instantaneous Hessian approximation (FDM)
    #             h = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
    #             # Difference of coefficient
    #             coef_diff += (gd*h)**2
    #         else:
    #             # Variance of gradient
    #             g_var = self.var_w*gd*gd + (1-self.var_w)*g_var
    #             # Instantaneous Hessian approximation (FDM)
    #             h = np.sqrt((coef_diff+self.eps_)/(g_var+self.eps_))
    #             # Difference of coefficient
    #             coef_diff = self.var_w*((gd*h)**2) + (1-self.var_w)*coef_diff
    #         # UPDATE and PROX
    #         sum_grad += gd
    #         if self.penalty == "l1":
    #             prox = 1 - self.alpha*Q/(np.abs(sum_grad)+self.eps_)
    #             coef_ = - h*sum_grad*prox*(prox > 0)
    #         else:
    #             coef_ = -h*sum_grad
    #         Q = 1/(np.abs(coef_)+self.eps_)
    #         Q = 1/(self.prop + len(coef_)*(1-self.prop)/Q.sum()*Q)
    #     return [coef_, [sum_grad, g_var, coef_diff]]
