import numpy as np

from scipy.sparse import csr_matrix

from .utility import Gradient, Base, BaseRegression


class AdaGradRdaClassifier(Base):
    """ADAGRAD RDA."""

    def __init__(self, loss="log", eta0=0.3, alpha=10**-4,
                 penalty='l1', fit_intercept=True, warm_start=False,
                 n_jobs=1):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)

    def _fit_bin(self, X, y, coef_, stats_=None):
        """stats_ is sum of gradient"""
        if stats_ is None:
            stats1_ = csr_matrix((1, X.shape[1]), dtype=np.float)
            stats2_ = csr_matrix((1, X.shape[1]), dtype=np.float)
        else:
            stats1_, stats2_ = stats_
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            stats1_ = stats1_.tolil() + gd.tolil()
            stats2_ = stats2_.tolil() + gd.power(2).tolil()
            coef_ = stats1_.copy()
            coef_[0, [coef_.indices]] = -self.eta0 * \
                coef_.data/stats2_[0, [coef_.indices]].toarray()[0]
            if self.penalty == "l1":
                prox = stats1_.copy()
                prox[0, [prox.indices]] = 1 - self.alpha*i/np.abs(prox.data)
                prox = prox.multiply(prox > 0)
                coef_ = coef_.multiply(prox)
        return [coef_, [stats1_, stats2_]]  # coef and sum of gradient


class AdaGradFbsClassifier(Base):
    """ADAGRAD ForwardBackwardSplittingType."""

    def __init__(self, loss="log", eta0=0.3, alpha=10**-4,
                 penalty='l1', fit_intercept=True, warm_start=False,
                 n_jobs=1):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)

    def _fit_bin(self, X, y, coef_, stats_=None):
        """stats_ is sum(variance of gradient)"""
        if stats_ is None:
            stats_ = csr_matrix((1, X.shape[1]), dtype=np.float)
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            stats_ = stats_.tolil()+gd.power(2).tolil()
            coef_ = coef_.tolil()-self.eta0*gd \
                         .multiply(csr_matrix((np.sqrt(1/stats_.data),
                                               stats_.indices,
                                               stats_.indptr),
                                              stats_.shape))
            del gd
            if self.penalty == "l1":
                prox = coef_.multiply(stats_.sqrt()) \
                       / (self.eta0*self.alpha+10**-6)
                prox[0, [prox.indices]] = 1 - 1/np.abs(prox.data)
                prox = prox.multiply(prox > 0)
                coef_ = coef_.multiply(prox)
                del prox
        return [coef_, stats_]


class AdaGradRdaRegressor(BaseRegression):
    """ADAGRAD RDA."""

    def __init__(self, loss="square", eta0=0.3, alpha=10**-4,
                 penalty='l1', fit_intercept=True, warm_start=False,
                 n_jobs=1):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)

    def _fit(self, X, y, coef_, stats_=None):
        """stats_ is sum of gradient"""
        if stats_ is None:
            stats1_ = np.zeros(X.shape[1])
            stats2_ = np.zeros(X.shape[1])
        else:
            stats1_, stats2_ = stats_
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            stats1_ += gd
            stats2_ += gd*gd
            coef_ = -self.eta0 * stats1_/(np.sqrt(stats2_)+10**-6)
            if self.penalty == "l1":
                prox = 1 - self.alpha*i/(np.abs(stats1_)+10**-6)
                coef_ = coef_ * prox * (prox > 0)
        return [coef_, [stats1_, stats2_]]  # coef and sum of gradient


class AdaGradFbsRegressor(BaseRegression):
    """ADAGRAD ForwardBackwardSplittingType."""

    def __init__(self, loss="square", eta0=0.3, alpha=10**-4,
                 penalty='l1', fit_intercept=True, warm_start=False,
                 n_jobs=1):
        self.set_params(loss=loss, eta0=eta0, alpha=alpha, penalty=penalty,
                        fit_intercept=fit_intercept, warm_start=warm_start,
                        n_jobs=n_jobs)

    def _fit(self, X, y, coef_, stats_=None):
        """stats_ is sum(variance of gradient)"""
        if stats_ is None:
            stats_ = np.zeros(X.shape[1])
        opt = Gradient(self.loss)
        for i in range(X.shape[0]):
            gd = opt.grad(X[i, :], y[i], coef_)
            stats_ += gd*gd
            coef_ += -self.eta0*gd/(np.sqrt(stats_)+10**-6)
            if self.penalty == "l1":
                prox = 1-self.eta0*self.alpha/(
                         np.abs(coef_)*np.sqrt(stats_)+10**-6)
                coef_ = coef_*prox*(prox > 0)
        return [coef_, stats_]
