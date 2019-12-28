import numpy as np
from .SGD import SGDClassifier, SGDRegressor


class FOBOSClassifier(SGDClassifier):
    """FOBOS."""

    def __init__(self, loss="log", eta0=0.1, power_t=0.5, alpha=10**-4,
                 fit_intercept=True, warm_start=False, n_jobs=1, eps_=10**-6):
        super().__init__(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                         penalty='l1', fit_intercept=fit_intercept,
                         warm_start=warm_start, n_jobs=n_jobs, eps_=eps_)


class FOBOSRegressor(SGDRegressor):
    """FOBOS."""

    def __init__(self, loss="square", eta0=0.1, power_t=0.5, alpha=10**-4,
                 fit_intercept=True, warm_start=False, n_jobs=1, eps_=10**-6):
        super().__init__(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                         penalty='l1', fit_intercept=fit_intercept,
                         warm_start=warm_start, n_jobs=n_jobs, eps_=eps_)
