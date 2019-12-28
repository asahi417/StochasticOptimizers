
"""Under Construction """
# 2重のmultiprocessingでエラー

import os
import numpy as np
import json
from datetime import datetime as dt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.base import clone
import multiprocessing as mp

from collections import OrderedDict

from .utility import create_log


def argwrapper(args):
    return args[0](**args[1])


def learning_batch(r, clf, X, y, multilabel, split):
    # - Constructs a new estimator with the same parameters.
    yy__, nz__ = np.array([]), np.array([])
    if multilabel:
        cv = KFold(n_splits=split, shuffle=True, random_state=r)
        ind = [i for _, i in cv.split(X)]
    else:
        cv = StratifiedKFold(n_splits=split, shuffle=True,
                             random_state=r)
        ind = [i for _, i in cv.split(X, y)]
    for i in range(len(ind)-1):
        clf.fit(X[ind[i]], y[ind[i]])
        yy__ = np.append(yy__,
                         1-clf.score(X[ind[i+1]], y[ind[i+1]]))
        nz__ = np.append(nz__, clf.coef_.count_nonzero())
    return yy__, nz__, clf


def LearningCurve01(X, y, classifiers,
                    rounds=1, split=10, path="", key=None, n_jobs=1):
    if key is None:
        path = path + dt.today().isoformat().replace(":", "-")
    else:
        path = path + key
    if not os.path.exists(path):
        os.makedirs(path)
    logger = create_log("%s/logger.log" % path)
    logger.info("Path: %s" % path)
    logger.info("X shape: %i, %i" % X.shape)
    logger.info("y shape: %i" % y.shape[0])
    logger.info("train size: %0.2f" % (y.shape[0]*(1-1/split)))
    if y.ndim > 1:
        multilabel = True
        logger.info("multilabel")
    else:
        multilabel = False
    yy, nz = OrderedDict(), OrderedDict()
    try:
        for name, clf in classifiers:
            logger.info(" training %s" % name)
            job = mp.Pool(n_jobs)
            func_args = [(learning_batch,
                          {'r': r, 'y': y, 'X': X,
                           'multilabel': multilabel, 'split': split,
                           'clf': clone(clf)})
                         for r in range(rounds)]
            re = job.map(argwrapper, func_args)
            yy_ = np.vstack([re[i][0] for i in range(rounds)])
            nz_ = np.vstack([re[i][1] for i in range(rounds)])
            yy[name] = {'mean': list(yy_.mean(0)), 'std': list(yy_.std(0))}
            nz[name] = {'mean': list(nz_.mean(0)), 'std': list(nz_.std(0))}
            joblib.dump(re[0][2], "%s/%s.pkl" % (path, name))
            job.close()
            job.join()
        logger.info(" save data")
        with open("%s/%s.json" % (path, "error"), "w") as f:
            json.dump(yy, f)
        with open("%s/%s.json" % (path, "nonzero"), "w") as f:
            json.dump(nz, f)
        with open("%s/%s.json" % (path, "simulation_setting"), "w") as f:
            json.dump({"split": split,
                       "data_classes": len(np.unique(y)),
                       "data_size": X.shape[0], "data_order": X.shape[1]}, f)
    except Exception as err:
            logger.exception("%s", err)
    return path
