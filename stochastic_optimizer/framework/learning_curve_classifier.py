import os
import numpy as np
import json

from scipy.sparse import csr
from datetime import datetime as dt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.base import clone

from collections import OrderedDict

from .utility import create_log


def LearningCurveClassifier(X, y, classifiers, rounds=1, split=10, path="",
                            key=None, test_size=0.3):
    """
     Parameters
    ----------
    X: input, ndarray, shape(sample, order)
    y: output, ndarray, shape(sample,), [0,1,...]
    classifiers: algorithms
        ex) [("ADAGRAD-rda", AdaGradRdaClassifier(eta0=0.3)),
             ("ADAGRAD-fbs", AdaGradFbsClassifier(eta0=0.1))]
    rounds: number of realizetion
    split: number of split (point of learning curve)
    path: (optinal) where to save the result
    key: (optinal) file name of save data

     Return
    ----------
    path to the saved result
    """

    if key is None:
        path = path + dt.today().isoformat().replace(":", "-")
    else:
        path = path + key
    if not os.path.exists(path):
        os.makedirs(path)
    logger = create_log("%s/logger.log" % path)
    logger.info("Path: %s" % path)
    logger.info("X shape: %i, %i" % X.shape)
    logger.info("y shape: %i" % len(y))
    logger.info("train size: %0.2f" % (len(y)*(1-1/split)))

    multilabel = True if y.ndim > 1 else False

    # mc: misclassification rate, yy: error rate for test data, nz: sparsity
    mc, yy, nz = OrderedDict(), OrderedDict(), OrderedDict()
    try:
        for name, clf in classifiers:
            logger.info(" training %s" % name)
            for r in range(rounds):
                logger.info("  rounds %i / %i" % (r+1, rounds))
                # Data Construction
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=test_size, random_state=r)
                _fold = {'n_splits': split, 'shuffle': True, 'random_state': r}
                cv = KFold(**_fold).split(X_train, y_train)
                ind = [i for _, i in cv]
                # - Constructs a new estimator with the same parameters.
                clf__ = clone(clf)
                for i in range(len(ind)):
                    logger.info("   iter %i / %i" % (i+1, split-1))
                    clf__.fit(X_train[ind[i]], y_train[ind[i]])
                    _yy = 1-clf__.score(X_test, y_test)
                    if i == 0:
                        _mc = _yy
                        mc__ = _mc
                        yy__ = _yy
                        nz__ = (clf__.coef_ != 0).sum()
                    else:
                        _mc += _yy
                        mc__ = np.append(mc__, _mc/(i+1))
                        yy__ = np.append(yy__, _yy)
                        nz__ = np.append(nz__, (clf__.coef_ != 0).sum())
                if r == 0:
                    mc_, yy_, nz_ = mc__, yy__, nz__
                else:
                    mc_ = np.vstack([mc_, mc__])
                    yy_ = np.vstack([yy_, yy__])
                    nz_ = np.vstack([nz_, nz__])
            if rounds == 1:
                mc[name] = {'mean': list(mc_), 'std': [0]*len(mc_)}
                yy[name] = {'mean': list(yy_), 'std': [0]*len(mc_)}
                nz[name] = {'mean': list(nz_.astype(np.float)),
                            'std': [0]*len(mc_)}
            else:
                mc[name] = {'mean': list(mc_.mean(0)), 'std': list(mc_.std(0))}
                yy[name] = {'mean': list(yy_.mean(0)), 'std': list(yy_.std(0))}
                nz[name] = {'mean': list(nz_.mean(0)), 'std': list(nz_.std(0))}
            joblib.dump(clf__, "%s/%s.pkl" % (path, name))

        logger.info(" save data")
        with open("%s/%s.json" % (path, "misclassification_rate"), "w") as f:
            json.dump(mc, f)
        with open("%s/%s.json" % (path, "error"), "w") as f:
            json.dump(yy, f)
        with open("%s/%s.json" % (path, "nonzero"), "w") as f:
            json.dump(nz, f)
        with open("%s/%s.json" % (path, "simulation_setting"), "w") as f:
            json.dump({"split": split,
                       "data_classes": len(clf__.classes_),
                       "rounds": rounds, "test_size": test_size,
                       "data_size": X.shape[0], "data_order": X.shape[1]}, f)
    except Exception as err:
            logger.exception("%s", err)
    return path
