# import numpy as np
#
# from .utility import Gradient, Base, BaseRegression
#
# import numpy as np
#
# from scipy.sparse import csr_matrix
#
# from .utility import Gradient, Base, BaseRegression
# from .SDA import SDAClassifier, SDARegressor
#
# """Framework of scikit learn.
#
#  Module
# ----------
# fit: training model
# predict: predict by learned model
#
#  Attributions
# ----------
# coef_: coefficient, ndarray, shape(class_n, order)
# stats1_: sum of gradient
# stats2_:
#
#  Parameters
# ----------
# loss: loss function
# eta0: initial learning rate
# power_t: decay rate for learning rate (constant if 0, max: 1)
#     eta_t = eta0/(t^power_t)
# alpha: coefficient for regularizer
# penalty: regularization penalty (l1, l2, None)
# warm_start: reuse preveous solution
# fit_intercept: boolean
# n_jobs: process number for multiprocessing
# M: memory size of BFGS
# L: compute Hessian statistics every L iterations
# b: size of SGD minibatch
# bH: size of Hessian minibatch
#
#  about algorithm
# ---------------------
# Hessian を計算する統計量 s_t, y_t を iteration の L 回毎に更新
# - 統計量 s_t, y_t を計算する時は過去 bH 分のデータを使用
# 実際に Hessian を計算する時は 過去 M 個分の s_t, y_t を用いる (L-BFGS)
# """
#
#
# class SQNRegressor(BaseRegression):
#     """Stochastic quasi-Newton
#     limited-memory BFGS."""
#
#     def __init__(self, loss="square", eta0=0.1, power_t=0.5, alpha=10**-4,
#                  penalty=None, fit_intercept=False, warm_start=False,
#                  n_jobs=1, M=10, L=10, b=1, bH=10, averaging=True):
#         self.set_params(loss=loss, eta0=eta0, power_t=power_t, alpha=alpha,
#                         penalty=penalty, fit_intercept=fit_intercept,
#                         warm_start=warm_start, n_jobs=n_jobs)
#         self.M = M
#         self.L = L
#         self.b = b
#         self.bH = bH
#         self.averaging = averaging
#
#     def _fit(self, X, y, coef_, stats_=None):
#         """
#         _b = 1 ~ b, minibatch for gradient
#         _t = 1 ~ L, hessian update cycle
#         _bH = 1 ~ bH, minibatch for hessian
#         _m = 1 ~ M, BFGS memory
#         """
#         opt = Gradient(self.loss)
#         _eps = 10**-6
#         _shape = X.shape[0]
#         if stats_ is None:
#             coef__ = np.zeros((self.L, X.shape[1]))
#             s_M = np.zeros((self.M, X.shape[1]))
#             y_M = np.zeros((self.M, X.shape[1]))
#             _t, _b, _m, ind = 0, 0, 0, 0
#             if self.averaging:
#                 grad__ = np.zeros((self.b, X.shape[1]))
#         else:
#             s_M, y_M = stats_[0], stats_[1]
#             _t, _b, _m, ind = stats_[2], stats_[3], stats_[4], stats_[5]
#             coef__ = stats_[6]
#             X = np.vstack([stats_[7], X])
#             if self.averaging:
#                 grad__ = stats_[8]
#
#         for i in range(_shape):
#             g_ = opt.grad(X[i+ind, :], y[i], coef_)
#             # minibatch for gradient
#             if self.averaging:
#                 grad__[_b, :] = g_
#                 if _b == self.b-1:
#                     _b = 0
#                 else:
#                     _b += 1
#                 g_ = grad__.sum(0)/np.min([i+1+ind, self.b])
#
#             # quasi-Newton update (two loop recursion)
#             if _m > 0:
#                 a, rho = np.zeros(_m), np.zeros(_m)
#                 for ii in range(_m):
#                     rho[ii] = y_M[_m-ii-1, :].dot(s_M[_m-ii-1, :])
#                     a[ii] = rho[ii]*s_M[_m-ii-1, :].dot(g_)
#                     g_ += -a[ii]*y_M[_m-ii-1, :]
#                 z = s_M[_m-1, :].dot(y_M[_m-1]) / \
#                     (y_M[_m-1].dot(y_M[_m-1])+_eps)*g_
#                 for ii in range(_m):
#                     z += s_M[ii, :]*(a[ii]-rho[ii]*y_M[ii, :].dot(s_M[ii, :]))
#                 coef_ += - self.eta0/pow(i+1+ind, self.power_t) * z
#             else:
#                 coef_ += - self.eta0/pow(i+1+ind, self.power_t) * g_
#
#             # Hessian update
#             coef__[_t, :] = coef_
#             if _t == self.L-1:
#                 w_ = coef__.sum(0)
#                 if i+ind != _t:
#                     # s
#                     s_ = (coef__.sum(0) - w_)/self.L
#                     w_ = coef__.sum(0)
#                     # y
#                     bH = np.min([self.bH, i+ind])
#                     y_ = np.vstack([opt.hessian_prod(s_, X[i+ind-_bH, :])
#                                     for _bH in range(bH)]).sum(0)/bH
#                     # store s, y
#                     if _m == self.M:
#                         s_M = np.vstack([s_, s_M[1:self.M, :]])
#                         y_M = np.vstack([y_, y_M[1:self.M, :]])
#                     else:
#                         s_M[_m, :] = s_
#                         y_M[_m, :] = y_
#                         _m += 1
#                 _t = 0
#             else:
#                 _t += 1
#
#             # if self.penalty == "l1":
#             #     prox = 1 - self.eta0 * \
#             #         self.alpha/(np.abs(coef_*pow(i+1, self.power_t))+10**-6)
#             #     coef_ = coef_ * prox * (prox > 0)
#
#         if self.averaging:
#             return [coef_, [s_M, y_M, _t, _b, _m, i+ind+1, coef__, X, grad__]]
#         else:
#             return [coef_, [s_M, y_M, _t, _b, _m, i+ind+1, coef__, X]]
