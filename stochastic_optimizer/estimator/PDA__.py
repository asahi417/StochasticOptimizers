import numpy as np

from scipy.sparse import csr_matrix

from .utility import Gradient, Base, BaseRegression
from .SDA import SDAClassifier, SDARegressor

"""Framework of scikit learn.

 Module
----------
fit: training model
predict: predict by learned model

 Attributions
----------
coef_: coefficient, ndarray, shape(class_n, order)
stats1_: sum of gradient
stats2_:

 Parameters
----------
loss: loss function
eta0: initial learning rate
power_t: decay rate for learning rate (constant if 0, max: 1)
    eta_t = eta0/(t^power_t)
alpha: coefficient for regularizer
penalty: regularization penalty (l1, l2, None)
warm_start: reuse preveous solution
fit_intercept: boolean
n_jobs: process number for multiprocessing
prop: proportionate metric ratio
    Q = I*prop + Q_*(1-prop)
"""


class PDAClassifier(SDAClassifier):
    """Projection-based Dual Averaging."""

    def __init__(self, eta0=0.1, alpha=10**-4, loss=None, power_t=None,
                 penalty='l1', fit_intercept=True, warm_start=False,
                 prop=None, n_jobs=1):
        super().__init__(loss="half-space", eta0=eta0, power_t=0, alpha=alpha,
                         penalty=penalty, fit_intercept=fit_intercept,
                         warm_start=warm_start, n_jobs=n_jobs)


class PDARegressor(BaseRegression):
    """Projection-based Dual Averaging."""

    def __init__(self, loss=None, eta0=0.1, power_t=None, alpha=10**-4,
                 penalty="l1", fit_intercept=True, prop=None,
                 warm_start=False, n_jobs=1):
        self.set_params(loss="hyper-plane", eta0=eta0, power_t=0, alpha=alpha,
                        penalty=penalty, fit_intercept=fit_intercept,
                        warm_start=warm_start, n_jobs=n_jobs)
        self.prop = prop

    def _fit(self, X, y, coef_, stats_=None):
        """stats_ is sum of gradient """
        if stats_ is None:
            stats_ = np.zeros(X.shape[1])
        opt = Gradient(self.loss)
        Q = 1
        for i in range(X.shape[0]):
            if self.prop is not None:
                Q = 1/(np.abs(coef_)+10**-6)
                Q = 1/(self.prop + len(coef_)*(1-self.prop)/Q.sum()*Q)
            stats_ += opt.grad(X[i, :], y[i], coef_, Q)
            if self.penalty == "l1":
                prox = 1 - self.alpha * Q / (np.abs(stats_)+10**-6)
                coef_ = - self.eta0 * stats_ * prox * (prox > 0)
            else:
                coef_ = -self.eta0 * Q * stats_
        return [coef_, stats_]  # coef and sum of gradient
