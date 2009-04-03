from recon import util
import numpy as np
## import pylab as P
from scipy import optimize
from time import time

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
    # actually have to compute full dk??
    #dk = np.zeros((N2,N2), 'D')
    dk = np.zeros((N2,N1,N2), 'D')
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
##     g1t_pos = UBPC.gtranslate(grad.gx, (tn-sig1))
##     k1t_pos = UBPC.gtranslate(grad.kx, (tn-sig1))*grad.gmaG0/(2*np.pi)
##     g1t_neg = UBPC.gtranslate(grad.gx, (Tl+tn-sig1))
##     k1t_neg = UBPC.gtranslate(grad.kx, (Tl+tn-sig1))*grad.gmaG0/(2*np.pi)
    g1t_pos = grad.gxt(tn-sig1)
    k1t_pos = grad.kxt(tn-sig1)*grad.gmaG0/(2*np.pi)
    g1t_neg = grad.gxt(tn-sig1)
    k1t_neg = grad.kxt(tn-sig1)*grad.gmaG0/(2*np.pi)
    
    ko[0] = np.sinc(n1p - (Lx*k1t_pos - a1*g1t_pos)[:,None])
    ko[1] = np.sinc(n1p - (Lx*k1t_neg - a1*g1t_neg)[:,None])


    if a2:
        # compute dk(n1,n2,n2') = del(n2,n2') - sinc(n2'-[n2-a2*g1t(n2,n1)])
        n2vec = np.arange(0.0, N2)
        n2pvec = np.arange(0.0, N2)
##         # subtract a2 cross term from n2vec
##         n2vec[0::2] = n2vec[0::2] - a2*g1t_pos[n1]
##         n2vec[1::2] = n2vec[1::2] - a2*g1t_neg[n1]
##         dk = -np.sinc(n2pvec - n2vec[:,None])
##         dk.flat[::N2+1] += 1.0
        dk[0::2] = -np.sinc(n2pvec[None,None,:] - n2vec[0::2,None,None] + \
                            a2*g1t_pos[None,:,None])
        dk[1::2] = -np.sinc(n2pvec[None,None,:] - n2vec[1::2,None,None] + \
                            a2*g1t_neg[None,:,None])
        for n2 in n2vec:
            dk[int(n2),:,int(n2)] += 1.0
        
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
        kpi[n2_0] = util.regularized_inverse(kpi[n2_0], l)
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
    # rows of the forward operator (Kpi*dK), problem shaped N2 x (N2'*N1')
    #t = time()
    Kfwd = np.zeros((N2, N2*N1), ko.dtype)
    kpi_rows = kpi[:,col,:].copy()
##     kpi_pos = kpi[0,col].copy()
##     kpi_neg = kpi[1,col].copy()
    
    #dk_col_pos = np.empty((N2*N1/2,), 'D')
    #dk_col_neg = np.empty((N2*N1/2,), 'D')
##     dk_col = np.empty((N2*N1/2), 'D')
##     for n2_0 in [0,1]:
##         dk_part = ko[n2_0,:,:][None,:,None,:] * dk[n2_0::2,:,:,None]
##         dk_part.shape = (N2*N1/2, N2*N1)
##         # k steps across columns of dK
##         kpi_row = kpi_rows[n2_0].copy()
##         for k in xrange(N1*N2):
##             n2p = k/N1
##             n1p = k%N2
##             t = time()
## ##             #      [ev, all-n1, n2p, n1p] * [all-n2, all-n1, n2p, n1p]
## ##             dk_col_pos = ko[0,:,n1p][None,:] * dk[0::2,:,n2p][:,:]
## ##             dk_col_neg = ko[1,:,n1p][None,:] * dk[1::2,:,n2p][:,:]
## ##             dk_col_pos.shape = (N1*N2/2,)
## ##             dk_col_neg.shape = (N1*N2/2,)
##             #dk_col[0] = dk_col_pos
##             #dk_col[1] = dk_col_neg
##             dk_col[:] = dk_part[:,k]
##             t_f[0] += time() - t
##             # i steps through rows of (Kpi*dK)
##             pol = 0
##             t = time()
##             for i in xrange(n2_0,N2,2):
##                 sl = slice(i/2*N1, (i/2+1)*N1)
##                 Kfwd[i,k] = np.dot(kpi_row, dk_col[sl])
##             t_f[1] += time() - t
##         del dk_part


    for n2 in xrange(N2):
        t = time()
        dk_part = (ko[n2%2,:,None,:] * dk[n2,:,:,None])
        dk_part.shape = (N1, N2*N1)
        Kfwd[n2] = np.dot(kpi_rows[n2%2], dk_part)
        #Kfwd[n2] = np.dot(dk_part.transpose(), kpi_rows[n2%2])
        t_f[0] += time() - t
        
    t = time()
    col_vec = np.dot(Kfwd, sig)
    sig.shape = (N2,N1)
    col_vec -= sig[:,col]
    col_vec *= a0_xterms_inv
    t_f[2] = time() - t
    
##     for n2_0 in [0,1]:
##         dKfwd[n2_0::2] = ko[n2_0,col][None,None,:] * dk[n2_0::2,:,None]
## ##     dKfwd = ko[:,col,None,:] * dk[:,:,None]
##     dKfwd.shape = (N2, N2*N1)
##     t_f[0] += time() - t

##     t = time()
##     rhs = np.dot(dKfwd, sig)
##     t_f[1] += time() - t

##     t = time()
##     for n2_0 in [0,1]:
##         # this should be adding a scalar to a segment of the rows
##         col_vec[n2_0::2] += np.dot(kpi[n2_0,col,:], rhs)
##     col_vec = a0_xterms_inv * col_vec
##     t_f[2] += time() - t

    print "kernel function creation: %1.2f"%(t_kern,)
    print "kernel inversion: %1.2f, application: %1.2f"%tuple(t_pi.tolist())
    print "forward op creation: %1.2f, application 1: %1.2f, application 2: %1.2f"%tuple(t_f.tolist())
    print "total time: %2.2f"%(t_kern + t_pi.sum() + t_f.sum(),)
    return col_vec

def eval_deghost_lite(epi, grad, coefs, chan, vol, r3,
                      l, col, constraints, mask):
    """Evaluate the ghost correction at a coef vector by projecting the
    ghosts onto a single column.
    """
    if not constraints(coefs):
        print 'out of bounds', coefs
        return 1e11*np.random.rand(1)[0]

    N2, N1 = epi.shape[-2:]
    if mask is None:
        col_mask = np.zeros(N2)
        col_mask[:N2/4] = 1.0
        col_mask[3*N2/4:] = 1.0
    else:
        if len(mask.shape) > 2:
            col_mask = mask[r3,:,col]
        elif len(mask.shape) > 1:
            col_mask = mask[:,col]
        else:
            col_mask = mask
        assert len(col_mask) == N2, 'provided mask is the wrong length'
    col_vec = deghost_lite(epi, grad, coefs, chan, vol, r3, l, col)
##     return col_vec*col_mask
    util.ifft1(col_vec, inplace=True, shift=True)
    g_nrg = ((np.abs(col_vec)*col_mask)**2).sum(axis=-1)
    print 'f(',coefs,') =',g_nrg
##     P.plot(np.abs(col_vec)/np.abs(col_vec).max())
##     P.plot(col_mask*.5, 'ro')
##     P.show()
    return g_nrg

## def get_grad(epi):
##     if epi.N1 == epi.n_pe and epi.fov_x > epi.fov_y:
##         epi.fov_y *= 2
##         epi.jsize = epi.isize
##     elif epi.fov_y == epi.fov_x and epi.n_pe > epi.N1:
##         epi.fov_y *= (epi.n_pe/epi.N1)
##         epi.jsize = epi.isize
##     print epi.fov_y
##     return UBPC.Gradient(epi.T_ramp, epi.T_flat, epi.T0,
##                          epi.n_pe, epi.N1, epi.fov_x)

def eval1D_wrap(x, idx, *args):
    # args[2] is an array of 5 coefs.. here just change args[2][0] to x
    args[2][idx] = x
    return eval_deghost_lite(*args)

def evalND_wrap(x, idc, *args):
    for i, idx in enumerate(idc):
        args[2][idx] = x[i]
    return eval_deghost_lite(*args)

def search_axes(epi, seeds, axes, cons=None, l=1.0,
                chan=0, vol=0, r3=0, mask=None):
    """Do an ND param search on given axes of the coeff vector
    """
    grad = util.grad_from_epi(epi)
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
    r = optimize.fmin(evalND_wrap, seed, args=(axes, epi, grad, coefs, chan,
                                               vol, r3, l, 64, cons, mask),
                      maxiter=75, full_output=1, disp=1,
                      ftol=1e-6, xtol=1e-5)
##     r = optimize.fmin_powell(evalND_wrap, seed, args=(axes, epi, grad, coefs,
##                                                       chan, vol, r3, l,
##                                                       64, cons, mask),
##                              maxfun=75., full_output=1, disp=1,
##                              ftol=1e-6, xtol=1e-5)
                      
    
    print r
    return r

def search_axis(epi, seeds, axis, l=1.0, chan=0, vol=0, r3=0):
    """Do a line search on a given axis of the coeff vector
    """
    grad = util.grad_from_epi(epi)

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
##     if validate:
##         #P.subplot(211)
##         P.plot(np.abs(rpos_fix), 'b', label='pos grad')
##         #P.plot(np.abs(rpos), 'b--')
##         P.plot(np.abs(rneg_fix), 'g', label='neg grad')
##         #P.plot(np.abs(rneg), 'g--')
## ##         P.subplot(212)
## ##         rneg = epi.cdata[chan,vol,sl,N2/2-1].copy()
## ##         rpos = epi.cdata[chan,vol,sl,N2/2].copy()
## ##         rneg_fix = util.regularized_solve(snc_neg, rneg, l)
## ##         rpos_fix = util.regularized_solve(snc_pos, rpos, l)
## ##         P.plot(np.abs(rpos_fix), 'b')
## ##         P.plot(np.abs(rpos), 'b--')
## ##         P.plot(np.abs(rneg_fix), 'g')
## ##         P.plot(np.abs(rneg), 'g--')
        
##         P.legend()
##         P.title('slice %d'%sl)
##         P.show()
    return invcorrcoef

def simple_sig1_line_search(epi, l=1, chan=0, vol=0):
    cons = lambda s1: s1[0]>=0.
    grad = util.grad_from_epi(epi)
    rlist = []
    coefs = np.zeros((epi.n_slice,))
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
        coefs[sl] = r[0]
    return coefs, rlist


## the search driver should find a ballpark sig1 first by a fast
## correlation optimization, and then move onto a projected ghost evaluation
## strategy in all four dimensions

## but what are the constraints?????

def search_coefs(epi, l=1.0, seeds=None, recon=True, mask=None):
    coefs = np.zeros((epi.n_chan, epi.n_slice, 4))
    # range over vol and slice, search all axes
    axes = range(4)
    rlist = []
    for c in range(epi.n_chan):
    #for c in [0]:
        if seeds is None:
            cf_s1 = simple_sig1_line_search(epi, l=l, chan=c)[0]
        for s in range(epi.n_slice):
        #for s in [0]:
            print "searching", (c, s)
            xseed = np.zeros(4)
            sig1_midpt = cf_s1[s] if seeds is None else seeds[c,s]
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
                xseed[0] = cf_s1[s]
                xseed[1] = .1
                xseed[2] = 1e-3
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

            r = search_axes(epi, xseed, axes, cons=cons,
                            l=l, chan=c, r3=s, mask=mask)
            rlist.append(r)
            coefs[c,s] = r[0]
            #open('coefs_%d_%d'%(c,s),'wb').write(r[0].tostring())
    return coefs, rlist
            
def feval_from_coefs(epi, coefs, l=1.0):
    grad = util.grad_from_epi(epi)
    cons = lambda v: True
    fvals = np.zeros((epi.n_chan, epi.n_slice), 'd')
    for c in range(epi.n_chan):
        for s in range(epi.n_slice):
           fvals[c,s] = eval_deghost_lite(epi, grad, coefs[c,s], c, 0,
                                          s, l, 64, cons)
    return fvals

## def plot_means(coefs):
##     cf_names = ['sig1', 'sig11', 'sig21', 'a0']
##     x_ax = np.arange(coefs.shape[1])
##     for axis in range(4):
##         P.subplot(400+10+(axis+1))
##         y_ax = coefs[:,:,axis].mean(axis=0)
##         y_er = coefs[:,:,axis].std(axis=0)
##         P.errorbar(x_ax, y_ax, y_er, elinewidth=1.5, fmt='ro')
##         P.plot(x_ax, y_ax, 'r--', linewidth=2, label='cross channel mean')
##         for c in range(12):
##             P.plot(coefs[c,:,axis], '-.')
##         if axis==3:
##             m,b,r = util.lin_regression(coefs[:,:,axis].mean(axis=0))
##             P.plot(np.arange(22)*m[0]+b[0], 'kD', label='linear fit')
##             P.plot(np.arange(22)*m[0]+b[0], 'k', linewidth=1)
##         P.gca().set_title(cf_names[axis])

##     foo = P.plot(coefs[c,:,axis], 'b-.', label='per channel trace')[0]
##     P.legend(loc=4)
##     foo.set_visible(False)
##     ylim = P.gca().get_ylim()
##     # make room for legend()
##     P.gca().set_ylim(-ylim[1]*.5, ylim[1])
##     P.show()
