import numpy as np
from sklearn.base import BaseEstimator


class Base(BaseEstimator):
    """Algorothm Framework
    - Scikit learn based
    """

    def preprocessing(self, X, y, coef_init, stats_init, class_n):
        """ Preprocessing data.
        (1) Check the data is sounds

        (2) Convert label data for classification
            - if y is converted such as
                  y = [[0,0,1], [1,0,0],...]
              then y.ndim have to be equall to class number
            - else, (y.ndim = 1)
                  y = [0,1,3,1,2,1,0,0] --> y = [[1,0,0,0], [0,1,0,0],..]

        (3) Initialize coefficient and statistics

         Parameters
        ----------
        X: input (ndarray, size: sample x order)
        y: output (ndarray)
            multilabel: shape(class, sample)
            else: shape(sample,)
        coef_init: initial coef_
        stats_init: initial stats_
        class_n: class number
            1 -> regression
            2 -> binary classification
            over 2 -> multiclass classification

         Return
        ----------
        X, y, coef_, stats_
        """
        multilabel = False
        # CHECK DATA
        if len(y) != len(X):
            raise ValueError("Length of 'y'(%i) and 'X'(%i) dont match"
                             % (len(y), len(X)))
        # Convert label data for classification
        if y.ndim == 1:
            if class_n > 2:
                y = np.vstack([(y == i).astype(int)*2-1
                               for i in self.classes_]).T
            elif class_n == 2:
                y = (y == self.classes_[1]).astype(int)*2-1
        else:
            multilabel = True
            if y.min() == 0:
                y = y*2-1
        # Add intercept
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        # Initialize coef_
        coef_ = self.ini_coef(coef_init, class_n, X.shape[1])

        # Initialize stats_
        stats_ = self.ini_stats(stats_init, class_n)
        return [X, y, coef_, stats_, multilabel]

    def ini_stats(self, stats_init, class_n):
        """Initialize stats_"""
        if self.warm_start and hasattr(self, "stats_"):
            return self.stats_
        elif self.warm_start and stats_init is not None:
            return stats_init
        else:
            if class_n <= 2:
                return None
            else:
                return [None]*class_n

    def ini_coef(self, coef_init, class_n, order):
        """Initialize coef_"""
        if self.warm_start and hasattr(self, "coef_"):
            return self.coef_
        else:
            # Regression and Bin classification
            if class_n in [1, 2]:
                if self.warm_start and coef_init is not None:
                    if len(coef_init) != order:
                        raise ValueError("coef init error")
                    else:
                        return coef_init
                else:
                    return np.zeros(order, dtype=np.float)
            # Multiclass classification
            else:
                if self.warm_start and coef_init is not None:
                    if coef_init.shape != (class_n, order):
                        raise ValueError("coef init error")
                    else:
                        return coef_init
                else:
                    return np.zeros((class_n, order), dtype=np.float)
