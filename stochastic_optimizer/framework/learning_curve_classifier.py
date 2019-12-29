import os
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.base import clone

from collections import OrderedDict

from .plot_curve import plot_curve
from ..log import create_log


def LearningCurveClassifier(X,
                            y,
                            classifiers,
                            rounds: int = 1,
                            split: int = 10,
                            test_size: float = 0.3,
                            path: str = "./"):
    """ Fit optimizer and accumulate historical errors for learning curve plot

     Parameters
    ----------
    X: input, ndarray, shape(sample, order)
    y: output, ndarray, shape(sample,), [0,1,...]
    classifiers: algorithms
        ex) [("ADAGRAD-rda", AdaGradRdaClassifier(eta0=0.3)),
             ("ADAGRAD-fbs", AdaGradFbsClassifier(eta0=0.1))]
    rounds: number of realizetion
    split: number of split (point of learning curve)
    path: path to save the result

     Return
    ----------
    path to the saved result
    """
    assert rounds >= 1
    assert split > 1
    assert 0.0 < test_size < 1.0

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    logger = create_log()
    logger.info("*** Learning Curve Test ***")
    logger.info("input shape: %i, %i" % X.shape)
    logger.info("output shape: %i" % len(y))
    logger.info("train size: %0.2f" % (len(y)*(1-1/split)))

    # mc: error rate, yy: error rate for test data, nz: sparsity
    mc, yy, nz = OrderedDict(), OrderedDict(), OrderedDict()
    try:
        for name, clf in classifiers:
            logger.info(" training %s" % name)
            for r in range(rounds):
                logger.info("  rounds %i / %i" % (r+1, rounds))
                # Data Construction
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=r)
                _fold = {'n_splits': split, 'shuffle': True, 'random_state': r}
                cv = KFold(**_fold).split(X_train, y_train)
                ind = [i for _, i in cv]
                # - Constructs a new estimator with the same parameters.
                clf__ = clone(clf)
                for i in range(len(ind)):
                    # logger.info("   iter %i / %i" % (i+1, split-1))
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
        # plot learning curve
        fig = plot_curve(
            yy,
            path=path,
            train_size=np.ceil(X.shape[0] * (1 - test_size)).astype(int),
            rounds=rounds,
            split=split)
        return fig
    except Exception as err:
        logger.exception("%s", err)
        return None
