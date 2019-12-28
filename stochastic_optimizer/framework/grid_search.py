from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.externals import joblib
from .. import create_log


def GridSearch(X,
               y,
               classifiers,
               scoring: str = "accuracy",
               path: str = None):
    """ Grid search hyperparameter
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

     Parameter
    ----------------
    scoring: "accuracy", "recall", "precision" for classification and
        "r2" for regression

    """
    logger = create_log()
    logger.info("*** start grid search (%s) ***" % str([n[0] for n in classifiers]))
    try:
        cv = ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        for name, clf_, params in classifiers:
            logger.info(" - searching: %s" % name)
            clf = GridSearchCV(clf_, params, scoring=scoring, cv=cv).fit(X, y)
            for k in clf.best_estimator_.get_params().keys():
                logger.info(" -> %s:%s" % (k, clf.best_estimator_.get_params()[k]))
            if path is not None:
                joblib.dump(clf, "%s/%s.pkl" % (path, name))
                logger.info("  the best model is saved at %s" % "%s/%s.pkl" % (path, name))
    except Exception as err:
            logger.exception("%s", err)

