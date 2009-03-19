import numpy as np
from IPython.kernel import client
from ghosting.set_up_mec import mec_setup

def simplex_step(func, xbar, fx0, xn, fxn, xp, fxp,
                 alpha, beta, gamma):
    """performs a simplex step, return new (x, fx)
    func : performs f(x)
    xbar : centroid of the N best points
    fx0 : f(x0)
    xn, fxn : f(xn) (xn == last best point)
    xp, fxp : f(xp) (xp == current point at this vertex)
    alpha, beta, gamma : simplex step parameters
    """
    xr = xbar + alpha*(xbar - xp)
##     print 'reflection:', xr
    fxr = func(xr)
    if fxr < fx0:
        # case 1
        #xe = xr + gamma*(xr - xbar)
        xe = xbar + gamma*alpha*(xbar - xp)
##         print 'expansion:', xe
        
        fxe = func(xe)
        if fxe < fxr:
            return 2, xe, fxe
        else:
            return 2, xr, fxr
    else:
        if fxr < fxn:
            # case 2
            return 1, xr, fxr
        else:
            best_pair = (xr, fxr) if fxr < fxp else (xp, fxp)
            xj = best_pair[0]
            xc = xbar - beta*(xbar - xj)
##             if fxr < fxp:
##                 print 'reflection contraction:', xc
##             else:
##                 print 'simplex contraction:', xc
            fxc = func(xc)
            if fxc < best_pair[1]:
                return 2, xc, fxc
            else:
                return (2,) + best_pair

def psimplex(x0, mec, xtol=1e-4, ftol=1e-4, maxiter=None,
             maxfun=None, full_output=0, debug=0):
    # assume that mec knows about the args and the code to run func
    P = len(mec.get_ids())
    mec.execute('def f(x): return func(x, *args)')
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

    all_fvals = []
    all_vecs = []
    # make step sizes 1% of each components relative contribution to the
    # vector length
    vlen = (x0**2).sum()**0.5
    #step_sizes = 0.05 * x0 / vlen
    step_sizes = np.ones_like(x0) * 0.05
    np.putmask(step_sizes, step_sizes==0, 0.001*vlen)
    alpha = 1; gamma = 2; beta = 0.5; tau = 0.5;
    mec.push(dict(alpha=alpha, gamma=gamma, beta=beta))
    simp = np.zeros((N+1, N), x0.dtype)
    simp[0] = x0
    print step_sizes
    for k in xrange(N):
        simp[k+1] = x0
        simp[k+1,k] = (1+step_sizes[k])*simp[k+1,k]

    fval = np.array(mec.map(func, simp))
    fcalls = N+1
    mec.push(dict(fcalls=fcalls/P))
##     for k in xrange(N+1):
##         print 'f(',simp[k],') =', fval[k]

    fval_p = fval[0]
    idx = np.argsort(fval)
    fval = fval[idx]
    simp = simp[idx]

    all_fvals.append(fval[0])
    all_vecs.append(simp[0])
    niter = 1
    fcalls = N+1
    while(niter < maxiter) and (fcalls < maxfun):
        if (np.abs(simp[1:]-simp[0]) < xtol).all() and \
           (np.abs(fval[1:]-fval[0]) < ftol).all():
            break

        xbar = simp[:-P].mean(axis=0)
        # xr is shaped (P, N)
        xr = xbar + alpha*(xbar - simp[-P:])

        mec.push(dict(xr=xr, xbar=xbar, fx0=fval[0], xn=simp[N-P],
                      fxn=fval[N-P]))

        for i in range(P):
##             mec.push(dict(xr=xr[i], xp=simp[N-P+i+1], fxp=fval[N-P+i+1]),
##                      targets=[i])
            mec.push(dict(xp=simp[N-P+i+1], fxp=fval[N-P+i+1]),
                     targets=[i])
##         print 'starting simplex step'
##         print 'centroid:', xbar, simp[-P:]
        mec.execute('n, x,fx = simplex_step(f, xbar, fx0, xn, fxn, xp, fxp, alpha, beta, gamma)')
        r = mec.get_result()
##         if debug:
##             try:
##                 print r[0]['stdout']
##             except:
##                 pass
        mec.execute('fcalls += n')
        x_steps = np.array(mec.pull('x'))
        fx = np.array(mec.pull('fx'))
##         print 'simplex results:'
##         for i in range(P):
##             print 'f(',x_steps[i],') = ',fx[i]

        if (x_steps==simp[-P:]).all():
            # shrink the simplex and evaluate all the new f(x)
            print 'shrinking the simplex'
            for k in xrange(1,N+1):
                #simp[k] = tau*simp[0] + (1-tau)*simp[k]
                simp[k] = simp[0] + tau*(simp[k]-simp[0])
            fval[1:] = np.array(mec.map(func, simp[1:]))
            mec.execute('fcalls += %d'%(N/P))
        else:
            for i in range(P):
                simp[N-P+i+1] = x_steps[i]
                fval[N-P+i+1] = fx[i]
        idx = np.argsort(fval)
        simp = simp[idx]
        fval = fval[idx]
        fval_p = fval[0]
        print 'f(',simp[0],') = ',fval[0]
        niter += 1
        fcalls = np.sum(mec.gather('fcalls'))
        all_fvals.append(fval[0])
        all_vecs.append(simp[0])
    x = simp[0]
    fval = fval[0]
    if niter >= maxiter:
        print 'Quit because number of iterations exceeded %d'%maxiter
    if fcalls >= maxfun:
        print 'Quit because number of function calls exceeded %d'%maxfun
    if full_output:
        return x, fval, all_fvals, all_vecs, niter, fcalls
    else:
        return x
        

def search_axes_simultaneous(seeds, axes, mec, job_idc):
    """performs ghosting coefficients search independently over P separate
    image planes. Only searches along coef axes specified in axes. Plane
    indices are indicated by job_idc, which is a grouping of 3-tuples
    ordered as (chan #, vol #, slice #)
    """
    coefs = np.array(seeds)
##     if cons is None:
##         def cons(vec):
##             return (vec >= 0).all()
    axes_seeds = np.take(coefs, axes, axis=-1)
    P = len(mec.get_ids())
    njobs = len(job_idc)
    assert P >= njobs, 'too many jobs for the # of processors'
    pending_jobs = []
    returns = []
    for i in xrange(njobs):
        mec.push(dict(chan=job_idc[i][0], vol=job_idc[i][1],
                      r3=job_idc[i][2], seed=axes_seeds[i],
                      coefs=coefs[i], axes=axes), targets=[i])
        print 'starting search on', job_idc[i]
        pj = mec.execute('r = fmin(evalND_full_wrap, seed, args=(axes, epi, grad, coefs, chan, vol, r3, l, cons, mask, cache), maxiter=150, full_output=1, disp=1, ftol=1e-6, xtol=1e-4)', targets=[i], block=False)
        pending_jobs.append(pj)
    mec.barrier(pending_jobs)
    for i in xrange(njobs):
        r = mec.pull('r', targets=[i])[0]
        returns.append(r)
    return returns
    
def simultaneous_sig1_line_search(mec, job_idc):
    cons = lambda s1: s1[0] >= 0.
    mec.push_function(dict(s1cons=cons))
    mec.execute('x = 0.5')
    njobs = len(job_idc)
    assert len(mec.get_ids()) >= njobs, 'too many jobs for the # of processors'
    rlist = []
    pending_jobs = []
    for i in xrange(njobs):
        mec.push(dict(chan=job_idc[i][0], vol=job_idc[i][1],
                      r3=job_idc[i][2]), targets=[i])
        pj = mec.execute('r = brent(corr_func, args=(epi, grad, chan, vol, r3, l, s1cons, 0), brack=(0,x,3), maxiter=500)', block=False)
        pending_jobs.append(pj)
    mec.barrier(pending_jobs)
    for i in xrange(njobs):
        rlist.append(mec.pull('r', targets=[i])[0])
    return rlist

def search_coefs_simultaneous(epi_path, gm_path='', l=1.0, seeds=None,
                              axes=None, status_str='', alt_indexing=None):
    if not axes:
        axes = range(4)
    mec = client.MultiEngineClient()
    P = len(mec.get_ids())
    def cons(vec):
        allpos = (vec >= 0).all()
        xterms_small = (vec[1:] < .25).all()
        return allpos and xterms_small
    
    mec_setup(mec, epi_path, gm_path, l=l, axes=axes,
              fmin_func='fmin', cons_func=cons)
    mec.push(dict(cache=None))
    allkeys = mec.keys()    
    for k in ('epi', 'grad', 'cons', 'l', 'mask', 'cache'):
        for mkeys in allkeys:
            assert k in mkeys, 'multi-engine client missing key: %s'%k    
    mec.execute('t = epi.n_chan, epi.n_vol, epi.n_slice')
    t = mec['t'][0]
    coefs = np.zeros(t+(4,), 'd')
    seed_mask = np.zeros(4)
    seed_mask[axes] = 1.
    if not alt_indexing:
        nd_idx = np.lib.index_tricks.ndindex(t)
        n_planes = nd_idx.total
    else:
        nd_idx = iter(alt_indexing)
        n_planes = len(alt_indexing)
        
    jobs_remain = True
    while jobs_remain:
        job_idc = []
        for i in xrange(P):
            try:
                job_idc.append(nd_idx.next())
            except StopIteration:
                jobs_remain = False
                
        if not job_idc:
            break
        if seeds is not None:
            seed_grp = [seeds[idx]*seed_mask for idx in job_idc]
        else:
            #seed_grp = some_other_seeding(job_idc)
            s1_seeds = simultaneous_sig1_line_search(mec, job_idc)
            seed_grp = [np.concatenate(([s1], np.random.rand(3)*.1))
                        for s1 in s1_seeds]
            seed_grp = [s*seed_mask for s in seed_grp]
                

        rlist = search_axes_simultaneous(seed_grp, axes, mec, job_idc)
    
        for i, idx in enumerate(job_idc):
            coefs[idx][axes] = rlist[i][0]
    return coefs
