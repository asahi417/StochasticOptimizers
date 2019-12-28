import numpy as np
import multiprocessing as mp
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError

from scipy.sparse import csr
from .base import Base


def argwrapper(args):
    return args[0](**args[1])


class BaseClassifier(Base, ClassifierMixin):
    """Framework of scikit learn.

     Module
    ----------
    fit: training model
    predict: predict by learned model

     Attributions
    ----------
    coef_: coefficient, ndarray, shape(sample, order)
    stats_: statistics for learning model depend on the optimizer
        ex) ADAGRAD -> variace of past gradient
            SDA -> sum of past gradient

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
    """

    def fit(self, X, y, coef_init=None, stats_init=None):
        """ Learning model.

         Parameters
        ----------
        X: input, ndarray, shape(sample, order)
        y: output, ndarray, shape(sample,)
        coef_init: initial coef_
        stats_init: initial stats_

         Return
        ----------
        self
        """

        # PREPROCESS
        if y.ndim == 1:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = [i for i in range(y.shape[1])]
        class_n = len(self.classes_)
        X = X.toarray() if csr.isspmatrix_csr(X) else X
        [X, y, self.coef_, self.stats_, self.multilabel] = \
            self.preprocessing(X, y, coef_init, stats_init, class_n)

        # LEARNING
        # - multiclass
        if class_n > 2 and self.n_jobs > 1:
            job = mp.Pool(self.n_jobs)
            func_args = [(self._fit,
                          {'X': X, 'y': y[:, n], 'coef_': self.coef_[n, :],
                           'stats_': self.stats_[n]})
                         for n, _ in enumerate(self.classes_)]
            re = job.map(argwrapper, func_args)
            self.coef_ = np.vstack([re[i][0] for i in range(class_n)])
            self.stats_ = [re[i][1] for i in range(class_n)]
            job.close()
            job.join()
        elif class_n > 2 and self.n_jobs == 1:
            for n, _ in enumerate(self.classes_):
                c, s = self._fit(X, y[:, n], self.coef_[n, :], self.stats_[n])
                self.coef_[n, :] = c
                self.stats_[n] = s
        # - binary
        else:
            [self.coef_, self.stats_] = self._fit(X, y, self.coef_,
                                                  self.stats_)
        return self

    def predict(self, X):
        """ Prediction.

         Parameters
        ----------
        X: input, ndarray, shape(sample, order)

         Return
        ----------
        prediction: ndarray, shape(sample,)"""

        if not hasattr(self, "coef_"):
            raise NotFittedError("This instance is not fitted yet")
        X = X.toarray() if csr.isspmatrix_csr(X) else X
        scores = self.decision_function(X)
        if self.multilabel:
            return (scores > 0).astype(np.int).T
        elif len(self.classes_) == 2:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(0)
        return self.classes_[indices]

    def score(self, X, y):
        if self.multilabel:
            s = (self.predict(X) == y).sum(1)
            s = (s == y.shape[1])
            return np.mean(s)
        else:
            return np.mean(self.predict(X) == y)

    def decision_function(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        return self.coef_.dot(X.T)

    # def predict_proba(self, X):
    #     """probability (soft max)"""
    #     if len(self.classes_) == 2:
    #         prob = 1/(1+np.exp(-self.decision_function(X)))
    #         return np.vstack([prob, 1-prob]).T
    #     else:
    #         prob = np.exp(self.decision_function(X))
    #         return prob/prob.sum(1).reshape(-1, 1)  # class x data
