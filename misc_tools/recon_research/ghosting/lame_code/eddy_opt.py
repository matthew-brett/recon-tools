from recon import util
from recon.operations import UBPC_siemens_1shot_new as UBPC
import numpy as np
import pylab as P
from scipy import optimize
from time import time
from recon.simulation import newghosts as ng
from eddy_opt_full import eval1D_full_wrap, search_axis_full

### THIS FILE --- PROJECTION (1 COLUMN) BASED GHOSTING OPTIMIZATION
### * 1D OPTIMIZATION OF SIGMA1, COUPLING TERMS (IN ORDER OF SIGNIFICANCE)
### * ND OPTIMIZATION OF COEFF VECTOR

def kernels_lite(image, grad, r3, n1, coefs):
    # only compute 2 rows of ko, since that fully describes the current setup
##     grad_polarity = 1
    N2 = image.n_pe
    N1 = image.N1
    # let a0 float
    sig1, sig11, sig21, a0 = coefs
    # if a0 is a range of numbers, return a list of a0_xterms to multiply
    # the data after the inverse problem
    if hasattr(a0, '__iter__'):
        a0_range = a0
    else:
        a0_range = [a0]
    ko = np.zeros((2,N1,N1), 'D')
    dk = np.zeros((N2,N2), 'D')
    a0_xterms_inv = np.zeros((len(a0_range), N2), 'D')
    
    
    Tf = float(image.T_flat); Tr = float(image.T_ramp); T0 = float(image.T0)
    Tl = 2*Tr + Tf
    #delT = (Tl-2*T0)/N1
    delT = (Tl-2*T0)/(N1-1)
    #delT = image.dwell_time/1e3
    Lx = image.fov_x; Ly = image.fov_y
    
    #a0 = grad.gmaG0*(sig01 + r3*sig31)/(2*np.pi)
    a1 = grad.gmaG0*Lx*sig11/(2*np.pi)
    a2 = grad.gmaG0*Ly*sig21/(2*np.pi)


    tn = np.arange(N1)*delT + T0
    n1p = np.arange(-N1/2, N1/2)
    # these are the only two gradient configurations
    g1t_pos = UBPC.gtranslate(grad.gx, (tn-sig1))
    k1t_pos = UBPC.gtranslate(grad.kx, (tn-sig1))*grad.gmaG0/(2*np.pi)
    g1t_neg = UBPC.gtranslate(grad.gx, (Tl+tn-sig1))
    k1t_neg = UBPC.gtranslate(grad.kx, (Tl+tn-sig1))*grad.gmaG0/(2*np.pi)
    ko[0] = np.sinc(n1p - (Lx*k1t_pos - a1*g1t_pos)[:,None])
    ko[1] = np.sinc(n1p - (Lx*k1t_neg - a1*g1t_neg)[:,None])
    # multiply a0 xterm to diagonal of posgrad matrix
    #ko[0].flat[0:N1*N1:(N1+1)] *= np.exp(1j*a0*g1t_pos)
    # multiply a0 xterm to diagonal of neggrad matrix
    #ko[1].flat[0:N1*N1:(N1+1)] *= np.exp(1j*a0*g1t_neg)

    if a2:
        #g1_xterm = np.empty(N2)
        #g1_xterm[0::2] = a2*g1t_pos[n1]
        #g1_xterm[1::2] = a2*g1t_neg[n1]
        n2vec = np.arange(0.0, N2)
        n2pvec = np.arange(0.0, N2)
        # subtract a2 cross term from n2vec
        n2vec[0::2] = n2vec[0::2] - a2*g1t_pos[n1]
        n2vec[1::2] = n2vec[1::2] - a2*g1t_neg[n1]
        dk = -np.sinc(n2pvec - n2vec[:,None])
        #dk = -np.sinc(n2pvec - (n2vec - g1_xterm)[:,None])
        dk.flat[::N2+1] += 1.0
##         for n2 in range(N2):        
##             g1t = g1t_pos if grad_polarity > 0 else g1t_neg
##             for n2p in range(N2):
##                 dk[n2,n2p] = -np.sinc(n2p - n2 + a2*g1t[n1])
##             dk[n2,n2] += 1.0        
##             grad_polarity *= -1
    if a0_range is not None:
        for i in range(len(a0_range)):
            a0_xterms_inv[i,0::2] = np.exp(-1j*a0_range[i]*g1t_pos[n1])
            a0_xterms_inv[i,1::2] = np.exp(-1j*a0_range[i]*g1t_neg[n1])
    return ko, dk, np.squeeze(a0_xterms_inv)
    

def deghost_lite(epi, grad, coefs, chan, vol, r3, l, col):
    """Applies to deghosting correction to one column
    """
    N2, N1 = epi.shape[-2:]
    if col < 0: col = N1/2
    t = time()
    ko, dk, a0_xterms_inv = kernels_lite(epi, grad, r3, col, coefs)
    
    t_kern = time() - t
    kpi = ko.copy()
    sig_slice = [slice(None)]*2
    k_pln = epi.cdata[chan, vol, r3].copy()
    t_pi = np.zeros(2)
    for n2_0 in [0,1]:
        sig_slice[-2] = slice(n2_0,N2,2)
        t = time()
        kpi[n2_0] = UBPC.regularized_inverse(kpi[n2_0], l)
        t_pi[0] += time() - t

        t = time()
        sig = k_pln[sig_slice].transpose().copy()
        sig = np.dot(kpi[n2_0], sig)
        sig = np.dot(ko[n2_0], sig)
        sig = -np.dot(kpi[n2_0], sig)
        t_pi[1] += time() - t

        k_pln[sig_slice] = sig.transpose()

    # make new copy of partially-corrected plane to operate on next
    sig_slice[-2] = slice(None)
    sig = k_pln.copy()
    col_vec = k_pln[:,col].copy()
    sig.shape = (N2*N1,)

    t_f = np.zeros(3)
    # now compute one column of the kspace slice by applying selected
    # rows of the forward operator dK, problem shaped N2 x (N2'*N1')
    t = time()
    dKfwd = np.empty((N2, N2, N1), ko.dtype)
    for n2_0 in [0,1]:
        dKfwd[n2_0::2] = ko[n2_0,col][None,None,:] * dk[n2_0::2,:,None]
##     dKfwd = ko[:,col,None,:] * dk[:,:,None]
    dKfwd.shape = (N2, N2*N1)
    t_f[0] += time() - t

    t = time()
    rhs = np.dot(dKfwd, sig)
    t_f[1] += time() - t

    t = time()
    for n2_0 in [0,1]:
        # this should be adding a scalar to a segment of the rows
        col_vec[n2_0::2] += np.dot(kpi[n2_0,col,:], rhs)
    col_vec = a0_xterms_inv * col_vec
    t_f[2] += time() - t

    print "kernel function creation: %1.2f"%(t_kern,)
    print "kernel inversion: %1.2f, application: %1.2f"%tuple(t_pi.tolist())
    print "forward op creation: %1.2f, application 1: %1.2f, application 2: %1.2f"%tuple(t_f.tolist())
    print "total time: %2.2f"%(t_kern + t_pi.sum() + t_f.sum(),)
    return col_vec

def eval_deghost_lite(epi, grad, coefs, chan, vol, r3, l, col, constraints):
    """Evaluate the ghost correction at a coef vector by projecting the
    ghosts onto a single column.
    """
    if not constraints(coefs):
        print 'out of bounds', coefs
        return 1e11

    N2, N1 = epi.shape[-2:]
    col_mask = np.zeros(N2)
    col_mask[:N2/4] = 1.0
    col_mask[3*N2/4:] = 1.0
    col_vec = deghost_lite(epi, grad, coefs, chan, vol, r3, l, col)
##     return col_vec*col_mask
    util.ifft1(col_vec, inplace=True, shift=True)
    g_nrg = ((np.abs(col_vec)*col_mask)**2).sum(axis=-1)
    print 'f(',coefs,') =',g_nrg
    return g_nrg

def get_grad(epi):
    if epi.N1 == epi.n_pe and epi.fov_x > epi.fov_y:
        epi.fov_y *= 2
        epi.jsize = epi.isize
    elif epi.fov_y == epi.fov_x and epi.n_pe > epi.N1:
        epi.fov_y *= (epi.n_pe/epi.N1)
        epi.jsize = epi.isize
    print epi.fov_y
    return UBPC.Gradient(epi.T_ramp, epi.T_flat, epi.T0,
                         epi.n_pe, epi.N1, epi.fov_x)

    
def sigma1(a1, grad):
    return a1 / (grad.gmaG0*grad.dx)

def eval1D_wrap(x, idx, *args):
    # args[2] is an array of 5 coefs.. here just change args[2][0] to x
    args[2][idx] = x
    return eval_deghost_lite(*args)

def evalND_wrap(x, idc, *args):
    for i, idx in enumerate(idc):
        args[2][idx] = x[i]
    return eval_deghost_lite(*args)


def search_axes(epi, seeds, axes, cons=None, l=1.0, chan=0, vol=0, r3=0):
    """Do an ND param search on given axes of the coeff vector
    """
    grad = get_grad(epi)
    coefs = np.array(seeds)
    # non-negative constraints
    if cons is None:
        def cons(vec):
            return (vec >= 0).all()
    seed = coefs[axes]
    print seed
    r = optimize.fmin(evalND_wrap, seed, args=(axes, epi, grad, coefs, chan,
                                               vol, r3, l, 64, cons),
                      maxiter=200, full_output=1, disp=1)
                      
    
    print r
    return r    

def search_axis(epi, seeds, axis, l=1.0, chan=0, vol=0, r3=0):
    """Do a line search on a given axis of the coeff vector
    """
    grad = get_grad(epi)

    coefs = np.asarray(seeds)
    # non-negative constraints
    def cons(vec):
        return (vec >= 0).all()
    
    seed = seeds[axis]
##     brack = optimize.bracket(eval1D_wrap, xa=0.0, xb=2*seed or 1,
##                              args=(axis, epi, grad, coefs, chan,
##                                    vol, r3, l, 64, constraints), maxiter=10)
##     print brack
                             
    r = optimize.brent(eval1D_wrap, args=(axis, epi, grad, coefs,
                                          chan, vol, r3, l, 64, cons),
                       brack = (0, 2*seed or 1),
                       maxiter=50,
                       full_output = 1
                       )
    print r
    return r


def corr_func(xv, epi, grad, chan, vol, sl, l, constraints, validate):    
    try:
        if len(xv) < 2:
            xv = np.array([xv[0], 0], 'd')
    except:
        xv = np.array([xv, 0])
    if not constraints(xv):
        #print 'out of bounds', coefs
        return 1e11
    Tr = epi.T_ramp
    Tf = epi.T_flat
    T0 = epi.T0
    Tl = 2*Tr + Tf
    N2 = epi.n_pe
    N1 = epi.N1

    #delT = (Tl-2*T0)/N1
    delT = (Tl-2*T0)/(N1-1)
    #delT = epi.delT/1e-6

    sig1 = xv[0]; sig11 = xv[1]
    a1 = grad.gmaG0*epi.fov_x*sig11/(2*np.pi)
    rneg = epi.cref_data[chan,vol,sl,0].copy()
    rpos = epi.cref_data[chan,vol,sl,1].copy()
    tn_neg = Tl + np.arange(128)*delT + T0
    tn_pos = np.arange(128)*delT + T0
    k_neg = UBPC.gtranslate(grad.kx, tn_neg - sig1) * grad.gmaG0/(2*np.pi)
    g_neg = UBPC.gtranslate(grad.gx, tn_neg - sig1)
    k_pos = UBPC.gtranslate(grad.kx, tn_pos - sig1) * grad.gmaG0/(2*np.pi)
    g_pos = UBPC.gtranslate(grad.gx, tn_pos - sig1)
    n1p = np.arange(-N1/2, N1/2)
    snc_pos = np.sinc(n1p - (k_pos*epi.fov_x - a1*g_pos)[:,None])
    snc_neg = np.sinc(n1p - (k_neg*epi.fov_x - a1*g_neg)[:,None])
    rpos_fix = UBPC.regularized_solve(snc_pos, rpos, l)
    rneg_fix = UBPC.regularized_solve(snc_neg, rneg, l)
    invcorrcoef = 1.0/(np.abs(rneg_fix) * np.abs(rpos_fix)).sum()
    print 'f(',xv,') =',invcorrcoef
    if validate:
        P.subplot(211)
        P.plot(np.abs(rpos_fix), 'b', label='pos grad')
        P.plot(np.abs(rpos), 'b--')
        P.plot(np.abs(rneg_fix), 'g', label='neg grad')
        P.plot(np.abs(rneg), 'g--')
        P.subplot(212)
        rneg = epi.cdata[chan,vol,sl,N2/2-1].copy()
        rpos = epi.cdata[chan,vol,sl,N2/2].copy()
        rneg_fix = UBPC.regularized_solve(snc_neg, rneg, l)
        rpos_fix = UBPC.regularized_solve(snc_pos, rpos, l)
        P.plot(np.abs(rpos_fix), 'b')
        P.plot(np.abs(rpos), 'b--')
        P.plot(np.abs(rneg_fix), 'g')
        P.plot(np.abs(rneg), 'g--')
        
        P.legend()
        P.title('slice %d'%sl)
        P.show()
    return invcorrcoef


def simple_sig1_sig11_search(epi, l=1, cons=None):
    grad = get_grad(epi)

    coefs = np.random.rand(epi.n_slice, 2)
    # suggest smaller values for sig11
    coefs[:,1] = .25*coefs[:,0]
    # non-negative constraints, also sig11 < sig1
    if cons is None:
        def cons(vec):
            return (vec>=0).all() and vec[1] < vec[0]
    #constraints = [lambda x: x>=0]*coefs.shape[1]
    #constraints = [lambda x: x>=-100]*coefs.shape[1]
    chan = 0; vol = 0
    rlist = []
    for sl in range(epi.n_slice):
        r = optimize.fmin(corr_func, coefs[sl], args=(epi, grad, chan, vol,
                                                      sl, l, cons, 0),
                          maxiter=500, full_output=1, disp=1)
        print r
        coefs[sl] = r[0]
        rlist.append(r)
                      
    return coefs, rlist

def simple_sig1_line_search(epi, l=1, chan=0, vol=0):
    cons = lambda s1: s1[0]>=0.
    grad = get_grad(epi)
    rlist = []
    coefs = np.zeros((epi.n_slice,2))
    for sl in range(epi.n_slice):
        x = .5
        r = optimize.brent(corr_func, args=(epi, grad, chan, vol,
                                            sl, l, cons, 0),
                           brack=(0,x,2),
                           maxiter=500,
                           full_output=1
                           )
        print
        rlist.append(r)
        coefs[sl,0] = r[0]
    return coefs, rlist
    

def validate_simple_sig1_sig11_search(epi, coefs, l=1, plot_fids=1,
                                      chan=0, vol=0):
    grad = get_grad(epi)
    inv_corrs = np.zeros(epi.n_slice)
    def cons(vec):
        return True
    for sl in range(epi.n_slice):
        inv_corrs[sl] = corr_func(coefs[sl], epi, grad, chan, vol, sl, l,
                                  cons, plot_fids)
    P.plot(inv_corrs)
        
    
