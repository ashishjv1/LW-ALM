from .lib import B, Bla, copy_tensor, to_tensor, size, transpose
from .lib_utils import get_perm

import sys
sys.path.append("../tensorly")
import tensorly as tl
from tensorly.kruskal_tensor import KruskalTensor, kruskal_normalise

import scipy
from scipy.io import loadmat
import numpy.linalg as la   #TODO: change to nla
import numpy as np


def cp_gradient(A, n, ns, N, I, R, Pmat, KRP_right0):

    right = B.arange(N-1, n, -1)
    left = B.arange(n-1, -1, -1)
    rsize = len(right)
    lsize = len(left)

    if n <= ns:
        if n == ns:
            if rsize == 1:
                KRP_right = A[right[0]]  
            elif rsize > 2:
                KRP_right, KRP_right0 = khatrirao(A, right, R, return_K2=True)
            elif rsize > 1:
                KRP_right = khatrirao(A, right, R)
            else:
                KRP_right = B.eye(1) 

            Pmat = B.reshape(Pmat, (-1, B.prod(I[right])))
            Pmat = Pmat @ KRP_right

        else:   
            Pmat = B.reshape(Pmat, (-1, I[right[-1]], R))
            if R > 1:
                Pmat = Pmat * A[right[-1]][None, :, :]
                Pmat = B.sum(Pmat, 1)
            else:
                Pmat = Pmat @ A[right[-1]]


        if lsize > 0:
            KRP_left = khatrirao(A, left, R)

            T = B.reshape(Pmat, (B.prod(I[left]), I[n], R))
            if R > 1:
                T = T * KRP_left[:, None, :]
                G = B.sum(T, 0)
            else:
                G = (KRP_left.T @ T).T
        else:
            G = B.reshape(Pmat, (-1, R))

    elif n >= ns+1:
        if n == ns+1:
            if lsize == 1:
                KRP_left = A[left].T
            elif lsize > 1:
                KRP_left = khatrirao_t(A, left, R)
            else:
                KRP_left = B.eye(1)

            T = B.reshape(Pmat, (B.prod(I[left]), -1))
            Pmat = KRP_left @ T

        else:
            if R > 1:
                Pmat = B.reshape(Pmat, (R, I[left[0]], -1))
                Pmat = Pmat * A[left[0]].T[:, :, None]
                Pmat = B.sum(Pmat, 1)
            else:
                raise NotImplementedError
                Pmat = B.reshape(Pmat, (I[left[0]], -1))
                Pmat = A[left[0]].T @ Pmat

        if rsize > 0:
            T = B.reshape(Pmat, (-1, I[n], B.prod(I[right])))
            if n == (ns+1) and rsize >= 2:
                if R > 1:
                    T = T * KRP_right0.T[:, None, :]
                    G = B.sum(T, 2).T
                else:
                    raise NotImplementedError
                    G = B.reshape(T, (-1, B.prod(I[right]))) @ KRP_right0
            else:
                KRP_right = khatrirao(A, right, R)
                if R > 1:
                    T = T * KRP_right.T[:, None, :]
                    G = B.sum(T, 2).T
                else:
                    raise NotImplementedError
                    G = B.reshape(T, (I[n], -1)) @ KRP_right
        else:
            G = B.reshape(Pmat, (R, -1)).T

    return G, Pmat, KRP_right0
    
def khatrirao(A, indices, R, return_K2=False):

    K = A[indices[0]]

    if not return_K2:
        for i in indices[1:]:
            K = A[i][:, None, :] * B.reshape(K, (1, -1, R))
        K = B.reshape(K, (-1, R))
        return K

    elif len(indices) > 2:
        for e, i in enumerate(indices[1:-1]):
            K = A[i][:, None, :] * B.reshape(K, (1, -1, R))

        K = B.reshape(K, (-1, R))
        K2 = copy_tensor(K)
        K = A[indices[-1]][:, None, :] * K[None, :, :]
        K = B.reshape(K, (-1, R))
        return K, K2

    else:
        raise ValueError

def khatrirao_t(A, indices, R):

    K = A[indices[0]].T

    for i in indices[1:]:
        K = A[i].T[:, :, None] * B.reshape(K, (R, 1, -1))
    
    K = B.reshape(K, (R, -1))
    
    return K

def fixsign(Ktensor):

    R = len(Ktensor.weights)
    N = len(Ktensor.factors)
    sng = B.empty((N, R))
    arange = B.arange(R)

    for u, U in enumerate(Ktensor.factors):
        indices = B.argmax(B.abs(U), 0)
        sng[u] = B.sign(U[indices, arange])

    negflag = (sng == -1)
    
    negindx = B.sum(negflag, 0) % 2 != 0

    for i in arange[negindx]:
        negflag[B.where(negflag[:, i])[0][-1], i] = False

    for U, flag in zip(Ktensor.factors, negflag):
        U[:, flag] = - U[:, flag]

    return Ktensor

def init(X_init, P_init, backend):

    tl.set_backend(backend)
#     print("Use {} backend".format(backend), file=sys.stderr)

    X = copy_tensor(X_init, dtype=B.float64)
    I = to_tensor(size(X))
    N = len(I)
    normX = Bla.norm(X)

    ns, p_perm = get_perm(I)
    #TODO: add complex support: IsReal = True
    
    # cp_init
    #TODO: make flag to avoid copy and make case with to_tensor
    U = [copy_tensor(U_i, dtype=B.float64) for U_i in P_init.factors]

    if p_perm is not None:
            I = I[p_perm]
            U = [U[p] for p in p_perm]
            X = transpose(X, p_perm)
      
    return X, U, I, N, normX, p_perm, ns 

def arange(Pi):

    P = kruskal_normalise(Pi)
    order = B.flip(B.argsort(P.weights), (0,))
    P.weights = P.weights[order]
    P.factors = [U[:, order] for U in P.factors]
    
    return P

def rearrange(U, p_perm):
    
    P = arange(KruskalTensor((None, U)))
    P = fixsign(P)

    if p_perm is not None:
        ipermute = B.argsort(p_perm)
        P.factors = [P.factors[p] for p in ipermute]

    return P

def kruskal_inner_product(kruskal_tensor1, kruskal_tensor2):
    weights1, factors1 = kruskal_tensor1
    weights2, factors2 = kruskal_tensor2
    norm = 1
    for factor1, factor2 in zip(factors1, factors2):
        norm *= B.dot(factor1.t(), factor2)
    
    norm = norm * (B.reshape(weights1, (-1, 1)) * B.reshape(weights2, (1, -1)))
    
    return B.sqrt(B.sum(norm))
