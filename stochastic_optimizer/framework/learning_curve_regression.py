# import os
# import numpy as np
# import json
# from datetime import datetime as dt
#
# from sklearn.model_selection import KFold
# from sklearn.externals import joblib
# from sklearn.base import clone
#
# from collections import OrderedDict
#
# from ..log import create_log
#
#
# def LearningCurveRegressor(X, y, regressors, t_coef=None,
#                            rounds=1, split=10, path="", key=None):
#     """
#      Parameters
#     ----------
#     X: input (csr_matrix, size: sample x order)
#     y: output (size: sample, [0,1,...])
#     classifiers: algorithms
#         ex) [("ADAGRAD-rda", AdaGradRdaClassifier(eta0=0.3)),
#              ("ADAGRAD-fbs", AdaGradFbsClassifier(eta0=0.1))]
#     t_coef: true coefficient for system mismatch
#     rounds: number of realizetion
#     split: number of split (point of learning curve)
#     path: (optinal) where to save the result
#     key: (optinal) file name of save data
#
#      Return
#     ----------
#     path to the saved result
#     """
#
#     if key is None:
#         path = path + dt.today().isoformat().replace(":", "-")
#     else:
#         path = path + key
#     if not os.path.exists(path):
#         os.makedirs(path)
#     logger = create_log("%s/logger.log" % path)
#     logger.info("Path: %s" % path)
#     logger.info("X shape: %i, %i" % X.shape)
#     logger.info("y shape: %i" % len(y))
#     logger.info("train size: %0.2f" % (len(y)*(1-1/split)))
#
#     mse, sm, nz = OrderedDict(), OrderedDict(), OrderedDict()
#     try:
#         for name, reg in regressors:
#             logger.info(" training %s" % name)
#             mse_, sm_, nz_ = np.array([[]]*(split-1)).T, \
#                 np.array([[]]*(split-1)).T, np.array([[]]*(split-1)).T
#             for r in range(rounds):
#                 logger.info("  rounds %i / %i" % (r+1, rounds))
#                 # - Constructs a new estimator with the same parameters.
#                 reg_ = clone(reg)
#                 mse__, sm__, nz__ = np.array([]), np.array([]), np.array([])
#                 cv = KFold(n_splits=split, shuffle=True, random_state=r)
#                 ind = [i for _, i in cv.split(X)]
#                 for i in range(len(ind)-1):
#                     logger.info("   iter %i / %i" % (i+1, split-1))
#                     reg_.fit(X[ind[i]], y[ind[i]])
#                     mse__ = np.append(mse__,
#                                       reg_.score(X[ind[i+1]], y[ind[i+1]]))
#                     if t_coef is not None:
#                         sm__ = np.append(sm__, reg_.score(t_coef, mesure="SM"))
#                     nz__ = np.append(nz__, (reg_.coef_ != 0).sum())
#                 mse_ = np.vstack([mse_, mse__])
#                 if t_coef is not None:
#                     sm_ = np.vstack([sm_, sm__])
#                 nz_ = np.vstack([nz_, nz__])
#             mse[name] = {'mean': list(mse_.mean(0)), 'std': list(mse_.std(0))}
#             if t_coef is not None:
#                 sm[name] = {'mean': list(sm_.mean(0)), 'std': list(sm_.std(0))}
#             nz[name] = {'mean': list(nz_.mean(0)), 'std': list(nz_.std(0))}
#             joblib.dump(reg, "%s/%s.pkl" % (path, name))
#         logger.info(" save data")
#         with open("%s/%s.json" % (path, "error"), "w") as f:
#             json.dump(mse, f)
#         with open("%s/%s.json" % (path, "nonzero"), "w") as f:
#             json.dump(nz, f)
#         if t_coef is not None:
#             with open("%s/%s.json" % (path, "sm"), "w") as f:
#                 json.dump(sm, f)
#         with open("%s/%s.json" % (path, "simulation_setting"), "w") as f:
#             json.dump({"split": split,
#                        "rounds": rounds,
#                        "data_size": X.shape[0], "data_order": X.shape[1]}, f)
#     except Exception as err:
#             logger.exception("%s", err)
#     return path
