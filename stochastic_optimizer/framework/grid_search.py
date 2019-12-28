import os
import numpy as np
import json
from datetime import datetime as dt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.externals import joblib

from .utility import create_log


def GridSearch(X, y, classifiers, scoring="accuracy",
               path="", key=None):

    """
     Parameter
    ----------------
    scoring: "accuracy", "recall", "precision" for classification and
        "r2" for regression
    """
    if key is None:
        path = path + dt.today().isoformat().replace(":", "-")
    else:
        path = path + key
    if not os.path.exists(path):
        os.makedirs(path)
    logger = create_log("%s/logger.log" % path)
    logger.info("Path: %s" % path)
    logger.info("Algorithm: %s" % str([n[0] for n in classifiers]))
    try:
        cv = ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        for name, clf_, params in classifiers:
            logger.info(" Grid searching: %s" % name)
            clf = GridSearchCV(clf_, params,
                               scoring=scoring, cv=cv).fit(X, y)
            logger.info("   Best fit")
            for k in clf.best_estimator_.get_params().keys():
                logger.info("     %s:%s"
                            % (k, clf.best_estimator_.get_params()[k]))
            joblib.dump(clf, "%s/%s.pkl" % (path, name))
    except Exception as err:
            logger.exception("%s", err)
    return path
