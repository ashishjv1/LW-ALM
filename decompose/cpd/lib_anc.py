from .lib_grad import cp_gradient, fixsign, init, rearrange, arange
from .lib_solver import qp_sphere_diagq
from .lib import *

import sys
sys.path.append("../tensorly")
import tensorly as tl
from tensorly.kruskal_tensor import KruskalTensor, kruskal_normalise

from tqdm import tqdm

import scipy
from scipy.io import loadmat
import numpy.linalg as la   #TODO: change to nla
import numpy as np

def cp_anc(X_init, R, delta, P_init=None, maxiter=10, tol=1e-6, verbose=False, linesearch=False, TraceFit=True, sqp_method="fzero", precision_tol=1e-12, max_stop_iter=10, iiii=-1):

    X, U, I, N, normX, p_perm, ns = init(X_init, P_init, backend) #maybe multiply last factor with lambda

    P = arange(KruskalTensor((P_init.weights, U)))
    lam = P.weights
    U = P.factors
    
    UtU = empty((N, R, R), dtype=B.float64)
    for n in range(N):
        UtU[n, :, :] = U[n].T @ U[n]

    if TraceFit:  fit = 0
    Pmat = None
    flagtol = 0
    normX2 = normX**2
    cost_local = [Bla.norm(lam)**2]
    
    mask = ones(N, dtype=B.bool)
    updateorder = list(range(ns, -1, -1)) + list(range(ns+1, N))

    for i in range(maxiter):

        if TraceFit: fittold = fit
        KRP_right0 = None

        for n in updateorder:
            
            U[n] *= lam
            
            # G = unfolding_dot_khatri_rao(X, KruskalTensor((None, U)), n)
            if n == ns or n == ns+1:
                G, Pmat, KRP_right0 = cp_gradient(U, n, ns, N, I, R, X, KRP_right0)
            else:
                G, Pmat, KRP_right0 = cp_gradient(U, n, ns, N, I, R, Pmat, KRP_right0)

            mask[n] = False
            Gamma = B.prod(UtU[mask, :, :], 0)
            mask[n] = True
            Gamma = (Gamma + Gamma.T) / 2

            eg, ug = eig(Gamma)
            eg_index = B.flip(B.argsort(eg), (0,))
            ug = ug[:, eg_index]
            eg = amax(eg[eg_index])

            b = G @ ug / B.sqrt(eg)
            deltan = delta**2 - normX2 + Bla.norm(b)**2

            if deltan < 0 :
                deltan = Bla.norm(U[n] @ ug * B.sqrt(eg) - b)**2            

            if deltan > 0:

                deltan = B.sqrt(deltan)
                b2 = b / eg
                b2n = Bla.norm(b2) 
                c = - b2 / b2n
                seg = 1 / eg
                s = deltan * (seg - seg[0]) / b2n + 1
                cn = - Bla.norm(c, None, 0)

                z, fval, lda2 = qp_sphere_diagq(s, cn, sqp_method, 0, precision_tol)
                
                z = c / (lda2 - s)
                xn = b - z * deltan
                U[n] = xn / B.sqrt(eg) @ ug.T

            if TraceFit and n == updateorder[-1]:
                innXXhat = B.sum(G * U[n])

            lam = Bla.norm(U[n], None, 0)
            U[n] /= lam
            UtU[n, :, :] = U[n].T @ U[n]
            cost_local += [Bla.norm(lam)**2]

        n0 = updateorder[0]
        U[n0] *= lam
        UtU[n0, :, :] = U[n0].T @ U[n0]
        lam = B.ones(R)

        if TraceFit:
            normresidual = B.sqrt(normX2 + B.sum(B.prod(UtU, 0)) - 2 * innXXhat)
            fit = 1 - (normresidual / normX)
            cost_diff = B.abs(cost_local[-1] - cost_local[-N-1]) / cost_local[-N-1]

            if i > 0 and cost_diff < tol:
                flagtol += 1
            else:
                flagtol = 0

            stop_flag = flagtol >= max_stop_iter
            # print("Iter {:2d}: |lambda|^2 = {:e}, diff = {:e}".format(i, cost_local[-1], cost_diff))

            if stop_flag:
                print("STOP!!!", i)
                break
    
    print("Final iter {:2d}: |lambda|^2 = {:e}, diff = {:e}".format(i, cost_local[-1], cost_diff))

    P = rearrange(U, p_perm)
    return P