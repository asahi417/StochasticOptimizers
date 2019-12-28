from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.preprocessing import LabelBinarizer


def argwrapper(args):
    return args[0](**args[1])


class OneVsRestClassifier(OneVsRestClassifier):

    def __init__(self, estimator, n_jobs=1):
        super().__init__(estimator, n_jobs)

    def fit(self, X, y):
        """Inherit sklearn 'OneVsRestClassifier'

        Modification: anable to warm_start with multilabel classification
        """

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        y_ = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in y_.T)

        # MODIFIED PART
        if hasattr(self, "estimators_"):
            # - warm coef and stats
            coefs, statss = [], []
            for e in self.estimators_:
                if e.warm_start and hasattr(e, "coef_"):
                    coefs.append(e.coef_)
                else:
                    coefs.append(None)
                if e.warm_start and hasattr(e, "stats_"):
                    statss.append(e.stats_)
                else:
                    statss.append(None)
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self.estimator.fit)(
                    X, column, coefs[i], statss[i])
                for i, column in enumerate(columns))
        else:
            # initialize coef and stats
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self.estimator.fit)(X, column) for column in columns)
        return self
