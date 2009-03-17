import numpy as np
from scipy.optimize.optimize import wrap_function
from IPython.kernel import client

from ghosting.search_driver_full_standalone import eval_deghost_full

def simple_eval_wrap(coefs, epi, grad, chan, vol, r3, l,
                     constraints, mask, cache):
    return eval_deghost_full(epi, grad, coefs, chan, vol, r3, l,
                             constraints, mask, cache)

def simplex_step(func, xr, xbar, fx0, xn, fxn, xp, fxp,
                 alpha, beta, gamma):
    """performs a simplex step, return new (x, fx)
    func : performs f(x)
    xr : initial reflection point
    xbar : centroid of the N best points
    fx0 : f(x0)
    xn, fxn : f(xn) (xn == last best point)
    xp, fxp : f(xp) (xp == current point at this vertex)
    alpha, beta, gamma : simplex step parameters
    """
    fxr = func(xr)
    if fxr < fx0:
        # case 1
        print 'case 1'
        xe = xr + gamma*(xr - xbar)
        fxe = func(xe)
        if fxe < fxr:
            return 2, xe, fxe
        else:
            return 2, xr, fxr
    else:
        if fxr < fxn:
            # case 2
            print 'case 2'
            return 1, xr, fxr
        else:
            print 'case 3'
            best_pair = (xr, fxr) if fxr < fxp else (xp, fxp)
            xj = best_pair[0]
            xc = beta*(xbar + xj)
            fxc = func(xc)
            if fxc < best_pair[1]:
                return 3, xc, fxc
            else:
                return (3,) + best_pair

def psimplex(x0, mec, xtol=1e-4, ftol=1e-4, maxiter=None,
             maxfun=None, full_output=0):
    # assume that mec knows about the args and the code to run func
    P = len(mec.get_ids())
    mec.execute('def f(x): return func(x, *args)')
    #mec.execute('fcalls, f = wrap_function(func, args)')
    func = mec.pull_function('f', targets=[0])[0]
    mec.push_function(dict(simplex_step=simplex_step))
    x0 = np.asfarray(x0).flatten()
    N = len(x0)
    rank = len(x0.shape)
    if not -1 < rank < 2:
        raise ValueError, "Initial guess must be a scalar or rank-1 sequence."
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200
    
    # make step sizes 1% of each components relative contribution to the
    # vector length
    vlen = (x0**2).sum()**0.5
    step_sizes = 0.01 * x0 / vlen
    np.putmask(step_sizes, step_sizes==0, 0.001*vlen)
    alpha = 1; gamma = 1; beta = 0.5; tau = 0.5;
    mec.push(dict(alpha=alpha, gamma=gamma, beta=beta))
    simp = np.zeros((N+1, N), x0.dtype)
    simp[0] = x0
    for k in xrange(N):
        simp[k+1] = x0
        simp[k+1,k] = step_sizes[k]

    #fval = np.array(tc.map(func, simp))
    fval = np.array(mec.map(func, simp))
    fcalls = N+1
    mec.push(dict(fcalls=fcalls/P))

    idx = np.argsort(fval)
    fval = fval[idx]
    simp = simp[idx]

        
    niter = 1
    fcalls = N+1
    while(niter < maxiter) and (fcalls < maxfun):
        if (np.abs(simp[1:]-simp[0]) < xtol).all() and \
           (np.abs(fsim[1:]-fsim[0]) < ftol).all():
            break

        xbar = simp[:-P].mean(axis=0)
        # xr is shaped (P, N)
        xr = xbar + alpha*(xbar - simp[-P:])

        mec.push(dict(xr=xr, xbar=xbar, fx0=fval[0], xn=simp[N-P-1],
                      fxn=fval[N-P-1]))

        for i in range(P):
            mec.push(dict(xr=xr[i], xp=simp[N-P+i], fxp=fval[N-P+i]),
                     targets=[i])
        print 'starting simplex step'
        mec.execute('n, x,fx = simplex_step(f, xr, xbar, fx0, xn, fxn, xp, fxp, alpha, beta, gamma)')
        mec.execute('fcalls += n')
        x_steps = np.array(mec.pull('x'))
        fx = np.array(mec.pull('fx'))
        print 'simplex results:'
        for i in range(P):
            print 'f(',x_steps[i],') = ',fx[i]

        if (x_steps==simp[-P:]).all():
            # shrink the simplex and evaluate all the new f(x)
            print 'shrinking the simplex'
            for k in xrange(1,N+1):
                simp[k] = tau*simp[0] + (1-tau)*simp[k]
            fval[1:] = np.array(mec.map(func, simp[1:]))
            mec.execute('fcalls += %d'%(N/P))
        else:
            for i in range(P):
                simp[N-P+i] = x_steps[i]
                fval[N-P+1] = fx[i]
        idx = np.argsort(fval)
        simp = simp[idx]
        fval = fval[idx]
        print 'f(',simp[0],') = ',fval[0]
        niter += 1
        fcalls = np.sum(mec.gather('fcalls'))
    x = simp[0]
    fval = fval[0]
    if niter >= maxiter:
        print 'Quit because number of iterations exceeded %d'%maxiter
    if fcalls >= maxfun:
        print 'Quit because number of function calls exceeded %d'%maxfun
    if full_output:
        return x, fval, iterations
    else:
        return x
        
        
