from .lib_grad import cp_gradient, fixsign, init, rearrange
from .lib import *

from tqdm import tqdm

import scipy
from scipy.io import loadmat
import numpy.linalg as la  #TODO: change to nla
import numpy as np

def cp_fast_als(X_init, R, P_init=None, maxiter=10, tol=1e-6, verbose=False, linesearch=False, TraceFit=True, max_stop_iter=10):

    X, U, I, N, normX, p_perm, ns = init(X_init, P_init, backend)

    UtU = empty((N, R, R), dtype=B.float64)
    for n in range(N):
        UtU[n, :, :] = U[n].T @ U[n]

    if TraceFit:  fit = 0
    flagtol = 0
    normX2 = normX**2

    mask = ones(N, dtype=B.bool)
    updateorder = list(range(ns, -1, -1)) + list(range(ns+1, N))

    for i in tqdm(range(maxiter)):
#     for i in range(maxiter):

        if TraceFit: fitold = fit
        KRP_right0 = None

        for n in updateorder:

            # G = unfolding_dot_khatri_rao(X, KruskalTensor((None, U)), n)                
            if n == ns or n == ns+1:
                G, Pmat, KRP_right0 = cp_gradient(U, n, ns, N, I, R, X, KRP_right0)
            else:
                G, Pmat, KRP_right0 = cp_gradient(U, n, ns, N, I, R, Pmat, KRP_right0)

            mask[n] = False
            A = B.prod(UtU[mask, :, :], 0)
            U[n] = solve(A, G.T).T
            mask[n] = True

            if TraceFit and n == updateorder[-1]:
                innXXhat = B.sum(G * U[n])

            lam = Bla.norm(U[n], None, 0) if i == 0 else mmax(U[n])
            U[n] = U[n] / lam
            UtU[n, :, :] = U[n].T @ U[n]
        
        U[0] = U[0] * lam  #TODO: may be line under this
        # U[updateorder[0]] = U[updateorder[0]] * lam  #TODO: maybe this line
        
        UtU[0, :, :] = U[0].T @ U[0]
        # UtU[:, :, updateorder[0]] = U[updateorder[0]].T @ U[updateorder[0]]
        # UtU(:,:,updateorder(1)) = conj(UtU(:,:,updateorder(1)))  #TODO: do we need it?

        stop_flag = False
        if TraceFit:
            normresidual = B.sqrt(normX2+B.sum(B.prod(UtU, 0))-2*innXXhat)
            fit = 1 - (normresidual / normX)
            fitchange = B.abs(fitold-fit)

            if i > 0 and fitchange < tol: # Check for convergence
                flagtol += 1
            else:
                flagtol = 0;

            stop_flag = flagtol >= max_stop_iter
            # print("Iter {:2d}: fit = {:e}, fitdelta = {:e}".format(i, fit, fitchange))

        if stop_flag:
            print("STOP!!!", i)
            break
    
    print("Final iter {:2d}: fit = {:e}, fitdelta = {:e}".format(i, fit, fitchange))

    P = rearrange(U, p_perm)
    
    return P