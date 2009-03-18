import search_driver_full_standalone as fullcorr
import search_driver_standalone as litecorr
import numpy as np
import pylab as P
from recon.simulation import newghosts as ng
from recon import util

def sigma1(a1, grad):
    return a1 / (grad.gmaG0*grad.dx)


def special_unwrap(phs, n=10):
    for i in range(n):
        pu = np.unwrap(phs[...,i::n])
        P.plot(np.arange(pu.shape[-1]), pu[0])
        phs[...,i::n] = pu
    P.show()
    return phs #np.unwrap(phs)

#def simple_unbal_phase_ramp(epi, chan, vol, sl):
def simple_unbal_phase_ramp(rdata, nramp, nflat, plot=False):
    #rdata = epi.cref_data[chan, vol, sl].copy()
    #nr = epi.n_ramp
    #nf = epi.n_flat
    N1 = rdata.shape[-1]
    rdata[:,:nramp+1] = 0.
    rdata[:,nflat+1:] = 0.
    #rev = rdata[0::2].copy()
    #rdata[0::2] = rev[...,::-1]
    util.ifft1(rdata, inplace=True, shift=True)

##     phs = np.angle(rdata)
##     phs = np.unwrap(np.angle(rdata))
    
##     height = phs[:,64]
##     height = ((height + np.sign(height)*np.pi)/(2*np.pi)).astype('i')
##     phs -= 2*np.pi*height[:,None]

    ref_funcs = rdata[:2].conjugate()*rdata[1:3]
    ref = ref_funcs[0]*ref_funcs[1].conjugate()
    ref_phs = np.unwrap(np.angle(ref))
    ht = ref_phs[N1/2]
    ht = int((ht + np.sign(ht)*np.pi)/(2*np.pi))
    ref_phs -= 2*np.pi*ht
##     P.plot(ref_phs)
##     P.show()

    #ref_funcs_phs = np.unwrap(np.angle(ref_funcs))
    ref_funcs_phs = np.angle(ref_funcs)
    ht = ref_funcs_phs[:,N1/2]
    ht = ((ht + np.sign(ht)*np.pi)/(2*np.pi)).astype('i')    
    ref_funcs_phs -= 2*np.pi*ht[:,None]

    pos_neg_diff = ref_funcs_phs[0]
    neg_pos_diff = ref_funcs_phs[1]
    
    rdata_peak = np.abs(ref).max()
    simple_q1_mask = np.where(np.abs(ref) > 0.1*rdata_peak, 1, 0)
    q1_pts = simple_q1_mask.nonzero()[0]

    var = (np.diff(ref_funcs_phs, n=2, axis=-1)**2).sum(axis=0)
    simple_var_mask = var < 1.5*np.median(var[q1_pts])
    q1_pts = np.intersect1d(q1_pts, 2+simple_var_mask.nonzero()[0])
    
    ref_phs = (pos_neg_diff-neg_pos_diff)
    #m,b,r = util.lin_regression(ref_phs/4, mask=simple_q1_mask)
    m,b,r = util.lin_regression(ref_phs/4, mask=simple_q1_mask)
    m = m[0]; b = b[0]
    #m,b,r = util.medfit(ref_phs[q1_pts]/4, x=q1_pts)
    if plot:
        P.plot(q1_pts, pos_neg_diff[q1_pts])
        P.plot(q1_pts, -neg_pos_diff[q1_pts])
        P.plot(q1_pts, ref_phs[q1_pts]/2)
        P.plot(q1_pts, (pos_neg_diff-neg_pos_diff)[q1_pts]/2, 'r--')
        P.plot(q1_pts, (2*m*np.arange(len(ref_phs))+2*b)[q1_pts], 'k--')
        P.show()
    return m


def compare_search_derived(epi, cf, chan):
    grad = litecorr.get_grad(epi)
    s1_der = np.zeros(epi.n_slice)
    for s in range(epi.n_slice):
        phs_ramp = simple_unbal_phase_ramp(epi, chan, 0, s)
        s1_der[s] = sigma1(phs_ramp, grad)
    P.plot(cf[chan,:,0], label='searched')
    P.plot(s1_der, label='derived')
    P.legend()
    P.show()
    
    
def simple_slice_recon(epi, grad, chan, vol, sl, coefs, l):
    k_pln = fullcorr.deghost_full(epi, grad, coefs, chan, vol, sl, l)
    epi.cdata[chan, vol, sl] = k_pln

def simple_volume_recon(epi, grad, chan, vol, coefs, l):
    assert epi.n_slice == coefs.shape[0]
    for sl in range(epi.n_slice):
        simple_slice_recon(epi, grad, chan, vol, sl, coefs[sl], l)
    epi.use_membuffer(chan)

def deghost_image(epi, grad, coefs, l):
    assert coefs.shape[0] == epi.n_chan
    for c in range(epi.n_chan):
        simple_volume_recon(epi, grad, c, 0, coefs[c], l)

def simple_volume_recon_planar(epi, chan, vol):
    Q3,N2,N1 = epi.shape[-3:]
    q1_ax = np.linspace(-N1/2., N1/2., N1, endpoint=False)
    n2sign = util.checkerline(N2)
    for s in range(Q3):
        s1 = simple_unbal_phase_ramp(epi, chan, vol, s)
        soln_pln = n2sign[:,None] * (s1*q1_ax)
        phs = np.exp(-1j*soln_pln)
        util.apply_phase_correction(epi.cdata[chan,vol,s], phs)
        #util.apply_phase_correction(epi.cref_data[chan,vol,s], phs[1:4])
    epi.use_membuffer(chan)

def drive_searches_full(epi, l=1.0):
    coefs = [0]*4
    a1 = ng.solve_coefs_l1(epi, pct=50.)[0]
    grad = litecorr.UBPC.Gradient(epi.T_ramp, epi.T_flat, epi.T0,
                                  epi.n_pe, epi.N1, epi.fov_x)
    s1_seed = sigma1(a1, grad)
    coefs[0] = s1_seed
    for ax in [0,1,2]:
        r = fullcorr.search_axis_full(epi, coefs, ax, l=l)
        coefs[ax] = r[0]
    return coefs

def drive_searches_short(epi, l=1.0, recon=True):
    allcoefs = []
    for sl in range(epi.n_slice):
        coefs = [0]*4
        a1 = simple_unbal_phase_ramp(epi, 0, 0, sl)
        grad = litecorr.get_grad(epi)
        s1_seed = sigma1(a1, grad)
        coefs[0] = s1_seed
        for ax in [0,1,2]:
            r = litecorr.search_axis(epi, coefs, ax, l=l, chan=0, vol=0, r3=sl)
            coefs[ax] = r[0]
        allcoefs.append(coefs)
    allcoefs = np.array(allcoefs)
    if recon:
        simple_volume_recon(epi, grad, 0, 0, allcoefs, l)
    return allcoefs

def drive_simple_search(epi, l=1, recon=True):
    chan = 0; vol = 0;
    def cons(vec):
        return (vec>=0).all() and vec[1] < .5*vec[0]
    coefs, _ = litecorr.simple_sig1_sig11_search(epi, l=l, cons=cons)
    if recon:
        grad = litecorr.get_grad(epi)
        allcoefs = np.zeros((epi.n_slice,4))
        allcoefs[:,0] = coefs[:,0]
        allcoefs[:,1] = coefs[:,1]
        simple_volume_recon(epi, grad, chan, vol, allcoefs, l)
    return coefs
    
def map_sig1_sig11_space(epi, sig1_range, sig11_range, sl=0):
    grad = litecorr.get_grad(epi)
    s1p = len(sig1_range)
    s11p = len(sig11_range)
    n_sl = epi.n_slice if sl < 0 else 1
    sl = 0 if sl < 0 else sl
    z = np.empty((n_sl, s1p, s11p), 'd')
    cons = lambda vec: True
    coefs = np.zeros(4)
    for s in xrange(n_sl):
        print "doing slice",s+sl
        for i in xrange(s1p):
            for j in xrange(s11p):
                xv = np.array([sig1_range[i], sig11_range[j]])
##                 z[s,i,j] = litecorr.corr_func(xv, epi, grad, 0, 0,
##                                               sl+s, 1.0, cons, 0)
##                 z[s,i,j] = litecorr.evalND_wrap(xv, [0, 1], epi, grad,
##                                                 coefs, 0, 0, sl+s, 1.0, 64,
##                                                 cons)
                z[s,i,j] = fullcorr.evalND_full_wrap(xv, [0, 1], epi, grad,
                                                          coefs, 0, 0, sl+s,
                                                          1.0, cons)
    i = np.indices((s1p, s11p))
    for s in range(n_sl):
        P.imshow(z[s])
        P.contour(i[0], i[1], z[s], 20, hold=True, cmap=P.cm.jet)
        P.title("slice=%d"%(sl+s))
        P.show()
    return z

def map_single_coef_space(epi, coef_range, axis, cf,
                          chan=0, sl=-1, mask=None, l=1.0):
    grad = util.grad_from_epi(epi)
    slices = xrange(epi.n_slice) if sl < 0 else [sl]
    z = np.zeros((len(slices), len(coef_range)))
    cf_bkp = cf.copy()
    vol = 0
    for s in slices:
        for i in xrange(len(coef_range)):
            args = (epi, grad, cf[0,s], chan, vol, s,
                    l, lambda x: True, mask, None)
            z[s,i] = fullcorr.eval1D_full_wrap(coef_range[i], axis, *args)
    cf[:] = cf_bkp
    return z
    

def map_coef_space(epi, ranges, chan=0, vol=0, sl=0):
    grad = litecorr.get_grad(epi)
    s1r, s11r, s21r, a0r = ranges
    s1p, s11p, s21p, a0p = map(len, ranges)
    z = np.empty((s1p, s11p, s21p, a0p), 'd')
    cons = lambda vec: True
    coefs = [0]*4
    for i in xrange(s1p):
        coefs[0] = s1r[i]
        for j in xrange(s11p):
            coefs[1] = s11r[j]
            for k in xrange(s21p):
                coefs[2] = s21r[k]
                coefs[3] = a0r
                z[i,j,k,:] = litecorr.eval_deghost_lite(epi, grad, coefs, chan,
                                                        vol, sl, 1.0, 64, cons)
    return z

def filter_fevals(fv):
    m = np.median(fv)
    return fv[(fv <= 1.25*m) & (fv > 0)]

def plot_fevals(fv):
    fvf = filter_fevals(fv)
    mn = fvf.min()
    P.plot(fvf)
    P.plot(mn*np.ones_like(fvf), 'r--')
    P.show()

def plot_increments(fv):
    fvf = fv[fv>0]
    pct = (fvf-fvf[0])/fvf[0]
    improvements = [pct[0]]
    where = [0]
    for p in xrange(1,len(pct)):
        if pct[p] < improvements[-1]:
            improvements.append(pct[p])
            where.append(p)
    P.plot(np.array(where), 100*np.array(improvements))
    P.plot(np.array(where), 100*np.array(improvements), 'bo')    
    P.show()


from scipy.optimize import brent, brentq, bracket, fminbound
def bracket_p(p, xi):
    l = np.concatenate((-p/xi, (.2-p[1:])/xi[1:]))
    #l = np.concatenate((-p/xi, (p[0]-p[1:])/xi[1:]))
    if (l[l<0] > -np.Inf).any():
        a = l[l<0].max()
    elif (l[l<=0] == -np.Inf).all():
        a = -5*np.dot(p,xi)
    else:
        a = 0.0 #because x+0v is always safe
    if (l[l>0] < np.Inf).any():
        b = l[l>0].min()
    elif (l[l>=0]==np.Inf).all():
        b = 5*np.dot(p,xi)
    else:
        b = 0.
    return a,b

def test_brack(p):
    n = 0
    eps = np.finfo(p.dtype).eps
    def cons(pp, p0):
        allpos = (pp>=-eps).all()
        #small = (pp[1:] <= .2+eps).all()
        small = (pp[1:] <= p0+eps).all()
        return allpos and small
    
    while n<100:
        xi = np.random.standard_normal(size=p.shape[0])
        a,b = bracket_p(p, xi)
        assert cons(p+a*xi, p[0]), 'lower bound failed'
        assert cons(p+b*xi, p[0]), 'upper bound failed'
        n += 1
    print "success"

def _linesearch_powell(func, p, xi, tol=1e-3):
    """Line-search algorithm using fminbound.

    Find the minimium of the function ``func(x0+ alpha*direc)``.

    """
    a,b = bracket_p(p, xi)
    def myfunc(alpha):
        return func(p + alpha * xi)

    print "minimizing over bracket", (a, b)
##     alpha_min, fret, iter, num = brent(myfunc, brack=(xa,xb,xc), full_output=1,
##                                        maxiter=20, tol=tol)
    alpha_min, fret, iter, num = fminbound(myfunc,a,b,xtol=tol,full_output=1)
    xi = alpha_min*xi
    return np.squeeze(fret), p+xi, xi

def wrap_function(function, args):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, function_wrapper

def fmin_powell(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
                maxfun=None, full_output=0, disp=1, retall=0, callback=None,
                direc=None):
    """Minimize a function using modified Powell's method.

    :Parameters:

      func : callable f(x,*args)
          Objective function to be minimized.
      x0 : ndarray
          Initial guess.
      args : tuple
          Eextra arguments passed to func.
      callback : callable
          An optional user-supplied function, called after each
          iteration.  Called as ``callback(xk)``, where ``xk`` is the
          current parameter vector.
      direc : ndarray
          Initial direction set.

    :Returns: (xopt, {fopt, xi, direc, iter, funcalls, warnflag}, {allvecs})

        xopt : ndarray
            Parameter which minimizes `func`.
        fopt : number
            Value of function at minimum: ``fopt = func(xopt)``.
        direc : ndarray
            Current direction set.
        iter : int
            Number of iterations.
        funcalls : int
            Number of function calls made.
        warnflag : int
            Integer warning flag:
                1 : Maximum number of function evaluations.
                2 : Maximum number of iterations.
        allvecs : list
            List of solutions at each iteration.

    *Other Parameters*:

      xtol : float
          Line-search error tolerance.
      ftol : float
          Relative error in ``func(xopt)`` acceptable for convergence.
      maxiter : int
          Maximum number of iterations to perform.
      maxfun : int
          Maximum number of function evaluations to make.
      full_output : bool
          If True, fopt, xi, direc, iter, funcalls, and
          warnflag are returned.
      disp : bool
          If True, print convergence messages.
      retall : bool
          If True, return a list of the solution at each iteration.


    :Notes:

        Uses a modification of Powell's method to find the minimum of
        a function of N variables.

    """
    # we need to use a mutable object here that we can update in the
    # wrapper function
    fcalls, func = wrap_function(func, args)
    x = np.asarray(x0).flatten()
    if retall:
        allvecs = [x]
    N = len(x)
    rank = len(x.shape)
    if not -1 < rank < 2:
        raise ValueError, "Initial guess must be a scalar or rank-1 sequence."
    if maxiter is None:
        maxiter = N * 1000
    if maxfun is None:
        maxfun = N * 1000


    if direc is None:
        direc = np.eye(N, dtype=float)
    else:
        direc = np.asarray(direc, dtype=float)

    fval = np.squeeze(func(x))
    x1 = x.copy()
    iter = 0;
    ilist = range(N)
    while True:
        fx = fval
        bigind = 0
        delta = 0.0
        for i in ilist:
            direc1 = direc[i]
            fx2 = fval
            print "doing line search in direction:", direc1
            fval, x, direc1 = _linesearch_powell(func, x, direc1, tol=xtol*100)
            if (fx2 - fval) > delta:
                delta = fx2 - fval
                bigind = i
        iter += 1
        if callback is not None:
            callback(x)
        if retall:
            allvecs.append(x)
        if (2.0*(fx - fval) <= ftol*(np.abs(fx)+np.abs(fval))+1e-20): break
        if fcalls[0] >= maxfun: break
        if iter >= maxiter: break

        # Construct the extrapolated point
        direc1 = x - x1
        x2 = 2*x - x1
        x1 = x.copy()
        fx2 = np.squeeze(func(x2))

        if (fx > fx2):
            print "doing extra line search in direction:", direc1
            t = 2.0*(fx+fx2-2.0*fval)
            temp = (fx-fval-delta)
            t *= temp*temp
            temp = fx-fx2
            t -= delta*temp*temp
            if t < 0.0:
                fval, x, direc1 = _linesearch_powell(func, x, direc1,
                                                     tol=xtol*100)
                direc[bigind] = direc[-1]
                direc[-1] = direc1
        print "final direction:", direc

    warnflag = 0
    if fcalls[0] >= maxfun:
        warnflag = 1
        if disp:
            print "Warning: Maximum number of function evaluations has "\
                  "been exceeded."
    elif iter >= maxiter:
        warnflag = 2
        if disp:
            print "Warning: Maximum number of iterations has been exceeded"
    else:
        if disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % iter
            print "         Function evaluations: %d" % fcalls[0]

    x = np.squeeze(x)

    if full_output:
        retlist = x, fval, direc, iter, fcalls[0], warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist


def matrix_l1norm(epi, grad, r3, coefs):
    N2, N1 = epi.shape[-2:]
    ko, dk, _ = fullcorr.kernels_full(epi, grad, r3, coefs)
    col_sums = np.zeros((N2*N1,), 'd')
    for n2 in xrange(N2):
        dk_part = ko[n2%2,:,None,:]*dk[n2,:,:,None]
        dk_part.shape = (N1,N2*N1)
        col_sums += np.abs(dk_part).sum(axis=0)
    return col_sums.max()
    
def matrix_linorm(epi, grad, r3, coefs):
    N2, N1 = epi.shape[-2:]
    ko, dk, _ = fullcorr.kernels_full(epi, grad, r3, coefs)
    row_sums = np.zeros((N2,N1), 'd')
    for n2 in xrange(N2):
        dk_part = ko[n2%2,:,None,:]*dk[n2,:,:,None]
        dk_part.shape = (N1,N2*N1)
        row_sums[n2] = np.abs(dk_part).sum(axis=-1)
    return row_sums.max()
