import numpy as np
import numpy.linalg as la
import importlib
from .lib import to_numpy, to_tensor
import warnings
# import sys

def f_lda(t, s, c):
    # t: scalar
    return ((c / (s - t))**2).sum() - 1

def g_lda(x, s, c):
    #TODO: CHECK! CHECK!
    c2 = c**2
    xs = x - s
    xs2 = xs**2
    f = np.sum(c2 / xs2, 0) - 1

    return f

def g_lda_prime(x, s, c):
    #TODO: CHECK! CHECK!
    c2 = c**2
    xs = x - s
    xs2 = xs**2
    g = -2 * np.sum(c2 / (xs2 * xs), 0)

    return g

def bound_lambda(c, s):
    c1 = np.abs(c[0])
    
    if 1 >= 1/np.abs(c[0]):
        #TODO: find how to solve this bad behavior
        lower_bound, upper_bound = 0, 0
        return lower_bound, upper_bound
    
    d = s[1] - s[0]
    p_4 = [c1**2, 2*c1*d, d**2-1, -2*c1*d, -d**2]
    t1 = np.roots(p_4)
    t1 = t1[np.abs(t1.imag)<1e-8].real
    t1[np.abs(1-t1)<1e-5] = 1
    
    t1 = np.min(t1[np.logical_and(t1>=1, t1<=1/c1)])
    lower_bound = 1 - c1 * t1
    
    d = s[-1] - s[0]
    p_4 = [c1**2, 2*c1*d, d**2-1, -2*c1*d, -d**2]
    t2 = np.roots(p_4)
    t2 = t2[np.abs(t2.imag)<1e-8].real
    t2[np.abs(1-t2)<1e-5] = 1
    
    t2 = np.max(t2[np.logical_and(t2>=1, t2<=1/c1)])
    upper_bound = 1 - c1 * t2
    
    return lower_bound, upper_bound
    
def bound_lambda_truncated(c, s):
    #TODO: CHECK!!! I played with s values previously!!!
    #TODO: +/- 1 for L
    L = np.where(s>2)[0][0] + 1
    L_min = np.min((20, L, np.round(s.size/10)))
    L = np.int32(np.max((2, L_min))) - 1
    solving_method = "fzero"
    verbosity=0

    # Lower bound
    cL = np.array(c[:L+1].tolist()+[la.norm(c[L+1:])])
    sL = s[:L+2]
    x_lw, fval_lw, lower_bound = solve_diagonal_qp(sL, cL, solving_method, verbosity)

    # Upper bound
    sL = np.array(s[:L+1].tolist()+[s[-1]])
    x_up, fval_up, upper_bound = solve_diagonal_qp(sL, cL, solving_method, verbosity)

    return lower_bound, upper_bound

def solve_diagonal_qp(s, c, solving_method="fzero",verbosity=0,precision_tol=1e-30):
    K = c.size
    if K == 1:
        lda = s[0]-c[0]
    else:
        if solving_method == "polyroot" and K < 10:
            raise NotImplementedError
        else:
            if s.size >= 2:
                if s[1]-s[0] < 1 and s[-1]-s[0] > 1 and s.size > 10:
                    lower_bound,upper_bound = bound_lambda_truncated(c, s)
                else:
                    lower_bound,upper_bound = bound_lambda(c, s)

                if lower_bound is None:  lower_bound = 1e-50
                if upper_bound is None:  upper_bound = 1-np.abs(c[0])
            else:
                lower_bound = 1e-50
                upper_bound = 1-np.abs(c[0])
                
            assert lower_bound > 0 and upper_bound >= lower_bound and upper_bound <= 1, "Bad bounds"

            lda_i = np.logspace(np.log10(lower_bound), np.log10(upper_bound), 20)
            lda_i = np.sort(lda_i.tolist()+[1-np.abs(c[0])])
            
            f_i = [f_lda(lda_ii, s, c) for lda_ii in lda_i]
            
            lda_i = np.array([0]+lda_i.tolist()+[1])
            f_i = np.array([-np.inf]+f_i+[np.inf])
            f_id = np.where(f_i>0)[0][0]
            lda_1 = lda_i[f_id-1]
            lda_2 = lda_i[f_id]

            if np.abs(lda_2-lda_1) <= 1e-15:
                lda = (lda_1 + lda_2) / 2
            else:
                if solving_method == "fzero":
                    lda = fzero(f_lda, (s, c), (lda_1, lda_2))
                        
                elif solving_method == "fsolve":                #TODO: try this branch or remove it
                    raise NotImplementedError
                    import scipy

                    lda_0 = (lda_1 + lda_2) / 2
                    solution = scipy.optimize.fsolve(g_lda, lda_0, args=(s, c), fprime=g_lda_prime)
                    if solution.size == 1:
                        lda = solution[0]
                    else:
                        assert False, solution

                elif solving_method == "fminbnd":
                    raise NotImplementedError
                
    x = c / (lda - s)
    fval = (lda + (c * x).sum()) / 2
    
    return  x, fval, lda

def qps_simplify(s, c, thresh_ident=1e-10):
    # This implementation differs from Matlab.
    c_size = c.size
    sd = np.diff(s)
    cid = np.where(np.abs(sd) > thresh_ident)[0] + 1
    cid = np.array([0]+cid.tolist()+[c_size])
    snew = s[cid[:-1]]
    c_len = cid.size - 1
    cnew = np.empty(c_len)
    
    A = np.zeros((c_size, c_len))
    for k in range(c_len):
        id1, id2 = cid[k], cid[k+1]
        cnew[k] = la.norm(c[id1:id2]) if id2 > id1 + 1 else c[id1]
        A[id1:id2,k] = c[id1:id2] / cnew[k]

    return snew, cnew, A    

def qp_sphere_diagq(s_init, b_init, solving_method="fzero", verbosity=0, precision_tol=1e-30):

    s = to_numpy(s_init)
    b = to_numpy(b_init)
    
    bn = la.norm(b)
    if bn == 0:        
        imin = np.argmin(s)
        s = s[imin]
        x = np.zeros_like(s)
        x[imin] = 1
        return x, s, 1

    K = s.size
    s_index = np.argsort(s)
    s = s[s_index]
    s = (s - s[0]) / bn + 1
    c = b[s_index] / bn
    s0, K0 = s, K
    
    s, c, Asimp  = qps_simplify(s, c)
    K = s.size
    zero_cid = np.abs(c) < 1e-12
    c[zero_cid] = 0
    
    if zero_cid.any():
        warnings.warn("Not tested yet. zero_cid.any()")
        nzero_cid = ~zero_cid
        s2 = s[nzero_cid]
        c2 = c[nzero_cid]

        if c[0] == 0:

            x = np.zeros(K)
            x[nzero_cid] = c2 / (s[0] - s2)
            normx2 = la.norm(x[nzero_cid])

            if normx2 < 1:

                warnings.warn("There may be two different solutions. Only one is used")
                x[0] = np.sqrt(1-normx2**2) # only one solution is used
                lda = 1
                fval = 0.5 * np.sum(s * x**2) + np.sum(c * x)

            else:

                if s2.size > 1:

                    s2 = s2 - s2[0] + 1
                    xreduced, fval, lda = solve_diagonal_qp(s2, c2, solving_method, verbosity, precision_tol)

                    x[0] = 0
                    nzero_cid[0] = False
                    x[nzero_cid] = xreduced

                    lda += s2[1] - 1
                    fval = 0.5 * np.sum(s * x**2) + np.sum(c * x)  #TODO: unchecked case

                else:
                    lda = s2 - c2
                    x[0] = 0
                    nzero_cid[0] = False
                    x[nzero_cid] = -1
                    fval = 0.5 * s + c

        else:

            x2, fval, lda = solve_diagonal_qp(s2, c2, solving_method, verbosity, precision_tol)
            x = np.zeros(K)
            x[nzero_cid] = x2

    else:
        x, fval, lda = solve_diagonal_qp(s, c, solving_method, verbosity, precision_tol)
    
    assert np.isscalar(lda), "error: solution in in form of array"
    
    fval = ((s * x**2).sum() / 2 + (c * x).sum()) * bn + (s0[0] - bn) / 2
    if K < K0:
        #TODO: it isn't tested as x isn't used further
        #warnings.warn("Not tested yet. K < K0")
        x = Asimp @ x[:, None] #.reshape(K, 1)
        x = x.reshape(-1)
        
    x = x[s_index]

    return to_tensor(x), fval, lda # array, float, float

def fzero(func, args, interval, tol=1e-15):
    a, b = interval
    fa, fb = func(a, *args), func(b, *args)
    
    assert fa * fb <= 0, "Values with the same sign"

    if fa == 0: return a
    if fb == 0: return b

    fc = fb
    while fb != 0 and a != b:
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a

        if np.abs(fc) < np.abs(fb):
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa

        m = 0.5 * (c - b)
        toler = 2.0 * tol * np.max((np.abs(b), 1.0))
        
        if np.abs(m) <= toler or fb == 0.0: 
            break

        if np.abs(e) < toler or np.abs(fa) <= np.abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            else:
                p = -p

            if 2.0 * p < 3.0 * m * q - np.abs(toler * q) and p < np.abs(e * q / 2):
                e = d
                d = p / q
            else:
                d = e = m

        a, fa = b, fb
        if np.abs(d) > toler:
            b = b + d
        elif b > c:
            b -= toler
        else:
            b += toler

        fb = func(b, *args)

    return b
