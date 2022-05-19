import numpy as np
from numpy.random import randn #, rand, randint
from itertools import combinations
from .lib import copy_tensor, to_numpy

import sys
sys.path.append("../tensorly")
import tensorly as tl
from tensorly.kruskal_tensor import KruskalTensor, kruskal_to_tensor


def cost_als(I):

    N = I.size
    ns_array = np.arange(np.ceil(N/2).astype(np.int32), N)
    ns_len = ns_array.size
    Allcomb = np.empty((ns_len, N), dtype=np.int32)
    mincost_array = np.empty(ns_len)
    arange = np.arange(N)
    mask = np.ones(N, dtype=np.bool_)

    for i, ns in enumerate(ns_array):

        allcomb = np.array(list(combinations(arange, ns)))
        temp = np.empty((len(allcomb), N), dtype=np.int32)

        for c, comb in enumerate(allcomb):
            temp[c, :ns] = comb
            mask[comb] = False
            temp[c, ns:] = arange[mask]
            mask[comb] = True
        
        Jn = np.cumprod(I[temp], axis=1)
        Kn = np.hstack((Jn[:, -1:], Jn[:, -1:] / Jn[:, :-1]))

        Tn = 0
        order = list(range(ns, 0, -1)) + list(range(ns+1, N+1))
        for n in order:
            if n == ns:
                Mn = Jn[:, 1:n-1].sum(axis=1) + np.min((Jn[:, n-1], Kn[:, n-1]), axis=0) + Jn[:, n+1:N].sum(axis=1) / Jn[:, n-1]
            elif n == ns + 1:
                Mn = Jn[:, 1:n-1].sum(axis=1) + (ns < N-1) * np.min((Jn[:, n-1], Kn[:, n-1]), axis=0) + Jn[:, n+1:N].sum(axis=1) / Jn[:, n-1]
            elif n < ns:
                Mn = Jn[:, 1:n+1].sum(axis=1) 
            elif n > ns + 1:
                Mn = Kn[:, n-2] + (n != N) * Kn[:, n-1] + Jn[:, n+1:N].sum(axis=1) / Jn[:, n-1]
            
            Tn += Mn

        idbest = np.argmin(Tn)
        mincost = Tn[idbest]
        bestord = temp[idbest, :N]
        Allcomb[i] = bestord
        mincost_array[i] = mincost

    n = np.argmin(mincost_array)

    return ns_array[n], Allcomb[n], mincost_array[n]


def get_perm(In):
    I = np.array(to_numpy(In))
    N = I.size

    if N < 14:
        permI = np.argsort(I)
        Is = I[permI]
        ns, bestord, _ = cost_als(Is)
        bestord = permI[bestord]
        p_perm = None
        assert (bestord < N).all()
        if (np.diff(bestord) < 0).any():
            p_perm = copy_tensor(bestord)

        ns -= 1
    
    else:
        p_perm = None
        if (np.diff(I) < 0).any():  # not sorted
            p_perm = copy_tensor(np.argsort(I))

        Jn = np.cumprod(I)
        Kn = np.hstack((Jn[-1], Jn[-1]/Jn[:-1]))
        ns = np.where(Jn<=Kn)[0][-1]

    return ns, p_perm
    