from recon import util
import eddy_corr_utils as eutils
import numpy as np
import pylab as P
from scipy import optimize
from time import time

def centered_ifft2(arr):
    chk = util.checkerboard(*arr.shape[-2:])
    return chk * np.fft.ifftn(chk*arr, axes=(-2,-1))

def kernels_full(image, grad, r3, coefs):
    sig1, sig11, sig21, a0 = coefs
    N2 = image.n_pe
    N1 = image.N1
    ko = np.zeros((2,N1,N1), 'D')
    dk = np.zeros((N2,N1,N2), 'D')
    a0_xterms_inv = np.zeros((N2,N1), 'D')
    
    Tf = float(image.T_flat); Tr = float(image.T_ramp); T0 = float(image.T0)
    Tl = 2*Tr + Tf
    delT = (Tl-2*T0)/(N1-1)
    Lx = image.fov_x; Ly = image.fov_y
    
    #a0 = grad.gmaG0*(sig01 + r3*sig31)/(2*np.pi)
    a1 = grad.gmaG0*Lx*sig11/(2*np.pi)
    a2 = grad.gmaG0*Ly*sig21/(2*np.pi)


    tn = np.arange(N1)*delT + T0
    n1p = np.arange(-N1/2, N1/2)

    # these are the only two gradient configurations
    # g(t) functions get hit with gamma/2PI * G0 when multiplied with a_i
    # k(t) functions should be hit with gamma/2PI * G0 now
##     g1t_pos = UBPC.gtranslate(grad.gx, (tn-sig1))
##     k1t_pos = UBPC.gtranslate(grad.kx, (tn-sig1))*grad.gmaG0/(2*np.pi)
##     g1t_neg = UBPC.gtranslate(grad.gx, (Tl+tn-sig1))
##     k1t_neg = UBPC.gtranslate(grad.kx, (Tl+tn-sig1))*grad.gmaG0/(2*np.pi)
    g1t_pos = grad.gxt(tn-sig1)
    k1t_pos = grad.kxt(tn-sig1)*grad.gmaG0/(2*np.pi)
    g1t_neg = grad.gxt(Tl+tn-sig1)
    k1t_neg = grad.kxt(Tl+tn-sig1)*grad.gmaG0/(2*np.pi)
    ko[0] = np.sinc(n1p - (Lx*k1t_pos - a1*g1t_pos)[:,None])
    ko[1] = np.sinc(n1p - (Lx*k1t_neg - a1*g1t_neg)[:,None])

    a0_xterms_inv[0::2] = np.exp(-1j*a0*g1t_pos)
    a0_xterms_inv[1::2] = np.exp(-1j*a0*g1t_neg)
    
    if a2:
        # calculate dk = del(n2,n2') - sinc[n2'-n2+a2*g1(t[n2,n1]-sig1)]
        # reduce g1(t[n2,n1]) to g1_pos(t[n1]) and g1_neg(t[n1])
        n2vec = np.arange(0.0,N2)
        n2pvec = np.arange(0.0,N2)
        dk[0::2] = -np.sinc(n2pvec[None,None,:] - n2vec[0::2,None,None] + \
                            a2*g1t_pos[None,:,None])
        dk[1::2] = -np.sinc(n2pvec[None,None,:] - n2vec[1::2,None,None] + \
                            a2*g1t_neg[None,:,None])
        for n2 in n2vec:
            dk[int(n2),:,int(n2)] += 1.0
            
    return ko, dk, a0_xterms_inv

def deghost_full(epi, grad, coefs, chan, vol, r3, l):
    N2, N1 = epi.shape[-2:]    
    t = time()
    ko, dk, a0_xterms_inv = kernels_full(epi, grad, r3, coefs)
    # if a2 is zero, then cut things short in the following code
    short_corr = (coefs[2]==0)
    t_kern = time() - t
    kpi = ko.copy()
    sig_slice = [slice(None)]*2
    k_pln = epi.cdata[chan, vol, r3].copy()
    t_pi = np.zeros(2)
    for n2_0 in [0,1]:
        sig_slice[-2] = slice(n2_0, N2, 2)
        t = time()
        kpi[n2_0] = util.regularized_inverse(kpi[n2_0], l)
        t_pi[0] += time() - t

        t = time()
        sig = k_pln[sig_slice].transpose().copy()
        sig = np.dot(kpi[n2_0], sig)
        # if a2 happens to be zero, we're done here.
        if short_corr:
            k_pln[sig_slice] = sig.transpose()
            continue
        
        sig = np.dot(ko[n2_0], sig)
        # Fixed per change in eq 28 (and appx C)
        #sig = -np.dot(kpi[n2_0], sig)
        sig = np.dot(kpi[n2_0], sig)
        t_pi[1] += time() - t

        k_pln[sig_slice] = sig.transpose()

    if short_corr:
        k_pln *= a0_xterms_inv
        return k_pln
    
    # make new copy of partially-corrected plane to operate on next
    sig_slice[-2] = slice(None)
    sig = k_pln.copy()
    #col_vec = k_pln[:,col].copy()
    sig.shape = (N2*N1,)

    t_f = np.zeros(3)
    for n2 in range(N2):
        #print "  n2:", n2
        # dK full is (N2*N1) x (N2'*N1')
        # ko is (2 x N1 x N1'), dk is (N2 x N2' x N1)
        t = time()
        #            n2,N1,N2',N1'     n2,N1,N2',N1'
        dK_part = ko[n2%2,:,None,:] * dk[n2,:,:,None]
        t_f[0] += time() - t
        
        dK_part.shape = (N1, N2*N1)
        
        t = time()
        rhs = np.dot(dK_part, sig)
        t_f[1] += time() - t
        
        sig_slice[-2] = n2
        # so .. image[...] is already S' = [(K+)(Ko)(K+)(S^T)]^T
        # just need to add [(K+)(dKp)(S'^T)]^T
        t = time()
        k_pln[sig_slice] += np.dot(kpi[n2%2], rhs).transpose()
        t_f[2] += time() - t

    k_pln *= a0_xterms_inv
##     print "kernel function creation: %1.2f"%(t_kern,)
##     print "kernel inversion: %1.2f, application: %1.2f"%tuple(t_pi.tolist())
##     print "forward op creation: %1.2f, application 1: %1.2f, application 2: %1.2f"%tuple(t_f.tolist())
##     print "total time: %2.2f"%(t_kern + t_pi.sum() + t_f.sum(),)
    return k_pln

def eval_deghost_full(epi, grad, coefs, chan, vol, r3,
                      l, constraints, mask, cache):
    """Evaluate the ghost correction at a coef vector.
    """
    if constraints is None:
        constraints = lambda x: True
    if not constraints(coefs):
        #print 'out of bounds', coefs
        print 'f(',coefs,') =',1e11
        return 1e11
    N2, N1 = epi.shape[-2:]
    if mask is None:
        ghost_mask = np.zeros((N2,N1))
        # unmask these regions
        ghost_mask[:N2/4,N1/4:3*N1/4] = 1.0
        ghost_mask[3*N2/4:,N1/4:3*N1/4] = 1.0
    else:
        if len(mask.shape) > 2:
            ghost_mask = mask[r3]
        else:
            ghost_mask = mask
        assert ghost_mask.shape == epi.cdata[chan,vol,r3].shape, 'provided mask is the wrong shape'

    k_pln = deghost_full(epi, grad, coefs, chan, vol, r3, l)

    #util.ifft2(k_pln, inplace=True, shift=True)
    k_pln = centered_ifft2(k_pln)
    g_nrg = ((np.abs(k_pln)*ghost_mask)**2).sum()
    print 'f(',coefs,') =',g_nrg
    if cache:
        cache.stash_cache(chan, r3, coefs, g_nrg)
    return g_nrg

def eval1D_full_wrap(x, idx, *args):
    # args[2] is a list of 4 coefs.. here just change args[2][idx] to x
    args[2][idx] = x
    return eval_deghost_full(*args)

def evalND_full_wrap(x, idc, *args):
    for i, idx in enumerate(idc):
        args[2][idx] = x[i]
    return eval_deghost_full(*args)

def search_axes_full(epi, grad, seeds, axes, cons=None, l=1.0,
                     chan=0, vol=0, r3=0, mask=None, cache=None):
    """Do an ND param search on given axes of the coeff vector
    """
    coefs = np.array(seeds)
    # non-negative constraints
    if cons is None:
        def cons(vec):
            return (vec >= 0).all()
    seed = coefs[axes]
    print seed
    # f(x) is usually bottoming out on the order of 1e-3.. so let's
    # make sure the optimization ftol is 1e-6
    # the values of x are from 1 to 1e-3 or 1e-4, so default xtol should be ok
    r = eutils.fmin(evalND_full_wrap, seed, args=(axes, epi, grad, coefs,
                                                    chan, vol, r3, l,
                                                    cons, mask, cache),
                      maxiter=150, full_output=1, disp=1, retall=1,
                      ftol=1e-6, xtol=1e-4)
##     r = eutils.fmin_powell(evalND_full_wrap, seed, args=(axes, epi, grad, coefs,
##                                                          chan, vol, r3, l,
##                                                          cons, mask, cache),
##                            maxfun=750., full_output=1, disp=1)
##                            #ftol=1e-6, xtol=1e-5)
                      
    
    print r
    return r

def search_axis_full(epi, grad, seeds, axis, l=1.0, cons=None,
                     chan=0, vol=0, r3=0, mask=None, cache=None):
    """Do a line search on a given axis of the coeff vector
    """
    grad = util.grad_from_epi(epi)
    try:
        axis = axis[0]
    except:
        pass
    coefs = np.asarray(seeds)
    # non-negative constraints
    if cons is None:
        def cons(vec):
            return (vec >= 0).all()
    
    seed = seeds[axis]
                             
    r = optimize.brent(eval1D_full_wrap, args=(axis, epi, grad, coefs,
                                               chan, vol, r3, l,
                                               cons, mask, cache),
                       brack = (0, 2*seed or 2),
                       maxiter=200,
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
        return np.random.rand(1)[0]*1e11
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
##     k_neg = UBPC.gtranslate(grad.kx, tn_neg - sig1) * grad.gmaG0/(2*np.pi)
##     g_neg = UBPC.gtranslate(grad.gx, tn_neg - sig1)
##     k_pos = UBPC.gtranslate(grad.kx, tn_pos - sig1) * grad.gmaG0/(2*np.pi)
##     g_pos = UBPC.gtranslate(grad.gx, tn_pos - sig1)
    k_neg = grad.kxt(tn_neg-sig1) * grad.gmaG0/(2*np.pi)
    g_neg = grad.gxt(tn_neg-sig1)
    k_pos = grad.kxt(tn_pos-sig1) * grad.gmaG0/(2*np.pi)
    g_pos = grad.gxt(tn_pos-sig1)
    n1p = np.arange(-N1/2, N1/2)
    snc_pos = np.sinc(n1p - (k_pos*epi.fov_x - a1*g_pos)[:,None])
    snc_neg = np.sinc(n1p - (k_neg*epi.fov_x - a1*g_neg)[:,None])
    rpos_fix = util.regularized_solve(snc_pos, rpos, l)
    rneg_fix = util.regularized_solve(snc_neg, rneg, l)
    invcorrcoef = 1.0/(np.abs(rneg_fix) * np.abs(rpos_fix)).sum()
    print 'f(',xv,') =',invcorrcoef
    if validate:
        #P.subplot(211)
        P.plot(np.abs(rpos_fix), 'b', label='pos grad')
        #P.plot(np.abs(rpos), 'b--')
        P.plot(np.abs(rneg_fix), 'g', label='neg grad')
        #P.plot(np.abs(rneg), 'g--')
##         P.subplot(212)
##         rneg = epi.cdata[chan,vol,sl,N2/2-1].copy()
##         rpos = epi.cdata[chan,vol,sl,N2/2].copy()
##         rneg_fix = util.regularized_solve(snc_neg, rneg, l)
##         rpos_fix = util.regularized_solve(snc_pos, rpos, l)
##         P.plot(np.abs(rpos_fix), 'b')
##         P.plot(np.abs(rpos), 'b--')
##         P.plot(np.abs(rneg_fix), 'g')
##         P.plot(np.abs(rneg), 'g--')
        
        P.legend()
        P.title('slice %d'%sl)
        P.show()
    return invcorrcoef

def simple_sig1_line_search(epi, grad, l=1, chan=0, vol=0, sl=0):
    cons = lambda s1: s1[0]>=0.
    #rlist = []
    #coefs = np.zeros((epi.n_slice,))
    #for sl in range(epi.n_slice):
    x = .5
    r = optimize.brent(corr_func, args=(epi, grad, chan, vol,
                                        sl, l, cons, 0),
                       brack=(0,x,3),
                       maxiter=500,
                       full_output=1
                       )
    coefs = r[0]
    return coefs


## the search driver should find a ballpark sig1 first by a fast
## correlation optimization, and then move onto a projected ghost evaluation
## strategy in all four dimensions

## but what are the constraints?????

def search_coefs(epi, l=1.0, seeds=None, mask=None,
                 axes=range(4), status_str=''):
    coefs = np.zeros((epi.n_chan, epi.n_slice, 4))
    # range over vol and slice, search indicated axes (default all)
    try:
        sfunc = search_axes_full if len(axes)>1 else search_axis_full
    except TypeError:
        sfunc = search_axis_full
    seed_mask = np.zeros(4)
    seed_mask[axes] = 1.
    rlist = []
    grad = util.grad_from_epi(epi)
    s_cache = SolnCache(epi.n_chan, epi.n_slice)
    for c in range(epi.n_chan):
        for s in range(epi.n_slice):
            print "searching", (c, s)
            if seeds is None:
                cf_s1 = simple_sig1_line_search(epi,grad,l=l,chan=c,sl=s)
            xseed = np.zeros(4)
            sig1_midpt = cf_s1 if seeds is None else seeds[c,s]
            def cons(vec):
                allpos = (vec >= 0).all()
                # bound sigma1 by +/- 5% -- could go tighter perhaps
                #s1_bounds = (sig1_midpt * .85, sig1_midpt * 1.15)
                #s1_bounds = (sig1_midpt * .95, sig1_midpt * 1.05)
                #s1_bounded = (vec[0]>s1_bounds[0] and vec[0]<s1_bounds[1])
                s1_bounded = True
                xterms_small = (vec[1:] < .2).all()
                return allpos and xterms_small and s1_bounded
            if seeds is None:
                xseed[0] = cf_s1
                xseed[1] = .1
                xseed[2] = np.random.rand(1) * .2 #1e-3
                xseed[3] = 5e-2
##                 try:
##                     # try to grab previous slice's points
##                     xseed[1:] = coefs[c,s-1,1:]
##                 except:
##                     # could try to grab previous channel's points
##                     pass
            else:
                xseed[:] = seeds[c,s] + np.random.normal(scale=1e-4, size=4)
                while not cons(xseed):
                    # try again until constraints are true
                    xseed[:] = seeds[c,s] + np.random.normal(scale=1e-4, size=4)
            # Zero out those points we're not searching for
            xseed *= seed_mask
            r = sfunc(epi, grad, xseed, axes, cons=cons,
                      l=l, chan=c, r3=s, mask=mask,
                      cache=s_cache)
                
            rlist.append(r)
            try:
                coefs[c,s] = r[0]
            except:
                coefs[c,s,axes] = r[0]
            if status_str:
                open(status_str+'_coefs_%d_%d'%(c,s), 'wb').write(r[0].tostring())
                #open('coefs_%d_%d'%(c,s),'wb').write(r[0].tostring())
    return coefs, rlist, s_cache

class inner_loop_section(object):
    def __init__(self, epi, grad, axes, cons, lmbda, job_idx,
                 gmask=None, seed=None, cache=None):
        self.epi = epi
        self.grad = grad
        self.axes = axes
        self.cons = cons
        self.lmbda = lmbda
        self.gmask = gmask
        self.seed = seed
        self.cache = cache
        self.chan, self.vol, self.sl = job_idx

    def run(self):
        # this basically finds the sigma 1 seed and runs the search over
        # given axes for (chan, vol, slice) in job_idx
        
        if self.seed is None:
            s1_cons = lambda x: x[0]>=0
            x = 0.5
            s1 = optimize.brent(corr_func, args=(self.epi, self.grad, self.chan,
                                                 self.vol, self.sl, self.lmbda,
                                                 s1_cons, 0),
                                brack=(0,x,3), maxiter=500.)
            self.seed = np.concatenate( ([s1], np.random.rand(3)*.1) )
        seed = np.asarray(self.seed)
        coefs = np.zeros_like(seed)
        coefs[self.axes] = seed[self.axes]
        seed = coefs[self.axes]
        r = eutils.fmin(evalND_full_wrap, seed, args=(self.axes, self.epi,
                                                      self.grad, coefs,
                                                      self.chan, self.vol,
                                                      self.sl, self.lmbda,
                                                      self.cons, self.gmask,
                                                      self.cache),
                        maxiter=150, full_output=1, disp=1, retall=1)
        return r[0]

            
def feval_from_coefs(epi, coefs, l=1.0, mask=None):
    grad = util.grad_from_epi(epi)
    cons = lambda v: True
    fvals = np.zeros((epi.n_chan, epi.n_slice), 'd')
    for c in range(epi.n_chan):
        for s in range(epi.n_slice):
           fvals[c,s] = eval_deghost_full(epi, grad, coefs[c,s], c, 0,
                                          s, l, cons, mask, None)
    return fvals

def plot_means(coefs):
    cf_names = ['sig1', 'sig11', 'sig21', 'sig31']
    x_ax = np.arange(coefs.shape[1])
    for axis in range(4):
        P.subplot(400+10+(axis+1))
        y_ax = coefs[:,:,axis].mean(axis=0)
        y_er = coefs[:,:,axis].std(axis=0)
        P.errorbar(x_ax, y_ax, y_er, fmt='ro')
        for c in range(12):
            P.plot(coefs[c,:,axis], '--')
        if axis==3:
            m,b,r = util.lin_regression(coefs[:,:,axis].mean(axis=0))
            P.plot(np.arange(22)*m[0]+b[0], 'kD')
        P.gca().set_title(cf_names[axis])
    P.show()

def energy_measures(epi, mask, chan=0):
    N2,N1 = epi.shape[-2:]
    epi.use_membuffer(chan)
    obj_mask = 1-mask
    obj_mask[:,:,:N1/4] = 0.
    obj_mask[:,:,3*N1/4:] = 0.
    ghost_nrg = ((np.abs(epi[:])*mask)**2).sum(axis=-1).sum(axis=-1)
    obj_nrg = ((np.abs(epi[:])*obj_mask)**2).sum(axis=-1).sum(axis=-1)
    sgr = obj_nrg/obj_mask.sum(axis=-1).sum(axis=-1)
    sgr /= ghost_nrg/mask.sum(axis=-1).sum(axis=-1)
    return ghost_nrg, obj_nrg, sgr

class SolnCache (object):
    def __init__(self, c, s):
        self.x_cache = [[[] for foo in range(s)] for foo2 in range(c)]
        self.feval_cache = [[[] for foo in range(s)] for foo2 in range(c)]
        
    def stash_cache(self, c, s, x, fx):
        self.feval_cache[c][s].append(fx)
        self.x_cache[c][s].append(x.tolist())

##     def return_arrays(self):
##         # return all x and f(x) for each (chan, slice)
        
def scache2array(sc):
    len1 = 0
    nc = len(sc.feval_cache); ns = len(sc.feval_cache[0])
    for i in xrange(nc):
        for j in xrange(ns):
            if len(sc.feval_cache[i][j]) > len1:
                len1 = len(sc.feval_cache[i][j])
    xarr = np.zeros((nc, ns, len1, 4))
    farr = np.zeros((nc, ns, len1))
    for i in xrange(nc):
        for j in xrange(ns):
            l1 = sc.x_cache[i][j]
            l2 = sc.feval_cache[i][j]
            xarr[i,j,:len(l1),:] = np.array(l1)
            farr[i,j,:len(l2)] = np.array(l2)
    return xarr, farr
