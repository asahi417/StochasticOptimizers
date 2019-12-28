import numpy as np
from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from .base import Base


class BaseRegressor(Base, RegressorMixin):
    """Framework of scikit learn.

     Module
    ----------
    fit: training model
    predict: predict by learned model

     Attributions
    ----------
    coef_: coefficient, ndarray, shape(class_n, order)
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
    n_jobs: process number for multiprocessing(currently no used in regression)
    """

    def fit(self, X, y, coef_init=None, stats_init=None):
        """ Learning model.

         Parameters
        ----------
        X: input, ndarray, shape(sample, order)
        y: output, ndarray, shape(sample,)
        coef_init: initial coef_, ndarray, shape(order, )
        stats_init: initial stats_

         Return
        ----------
        self
        """

        # PREPROCESS
        [X, y, self.coef_, self.stats_] = \
            self.preprocessing(X, y, coef_init, stats_init, 1)
        # LEARNING
        [self.coef_, self.stats_] = self._fit(X, y, self.coef_, self.stats_)
        return self

    def predict(self, X):
        """ Prediction.

         Parameters
        ----------
        X: input, ndarray, shape(sample, order)

         Return
        ----------
        predicted values, ndarray, shape(sample,) """

        if not hasattr(self, "coef_"):
            raise NotFittedError("This instance is not fitted yet")

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X.dot(self.coef_)

    def score(self, X, y=None, mesure=None):
        if mesure is None or "MSE":
            return np.mean((self.predict(X) - y)**2)
        elif mesure is "SM":  # X is true coef
            return np.mean((self.coef_-X)**2/X.dot(X))
