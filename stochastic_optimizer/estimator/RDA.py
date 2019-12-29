from .SDA import SDAClassifier, SDARegressor


class RDAClassifier(SDAClassifier):
    """Reguralized Dual Averaging."""

    def __init__(self, loss="log", eta0=0.1, alpha=10**-4,
                 fit_intercept=True, warm_start=False, n_jobs=1, eps_=10**-6):
        penalty = 'l1'
        power_t = 0.5
        super().__init__(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                         penalty=penalty, fit_intercept=fit_intercept,
                         warm_start=warm_start, n_jobs=n_jobs, eps_=eps_)


class RDARegressor(SDARegressor):
    """Reguralized Dual Averaging."""

    def __init__(self, loss="square", eta0=0.1, alpha=10**-4,
                 fit_intercept=True, warm_start=False, n_jobs=1):
        penalty = 'l1'
        power_t = 0.5
        super().__init__(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
                         penalty=penalty, fit_intercept=fit_intercept,
                         warm_start=warm_start, n_jobs=n_jobs, eps_=eps_)
