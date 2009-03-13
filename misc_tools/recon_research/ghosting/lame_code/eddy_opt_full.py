from recon import util
from recon.operations import UBPC_siemens_1shot_new as UBPC
import numpy as np
from scipy import optimize
from time import time

def kernels(image, grad, r3, coefs):
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
    print Lx, Ly
    
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

    a0_xterms_inv[0::2] = np.exp(-1j*a0*g1t_pos)
    a0_xterms_inv[1::2] = np.exp(-1j*a0*g1t_neg)
    
    if a2:
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
    ko, dk, a0_xterms_inv = kernels(epi, grad, r3, coefs)
    t_kern = time() - t
    kpi = ko.copy()
    sig_slice = [slice(None)]*2
    k_pln = epi.cdata[chan, vol, r3].copy()
    t_pi = np.zeros(2)
    for n2_0 in [0,1]:
        sig_slice[-2] = slice(n2_0, N2, 2)
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
    #col_vec = k_pln[:,col].copy()
    sig.shape = (N2*N1,)

    t_f = np.zeros(3)
    for n2 in range(N2):
        #print "  n2:", n2
        # dK full is (N2*N1) x (N2'*N1')
        # ko is (N2 x N1 x N1'), dk is (N2 x N2' x N1)
        #            N2,N1,N2',N1'     N2,N1,N2',N1'
        t = time()
        dK_part = ko[n2%2,:,None,:] * dk[n2,:,:,None]
        t_f[0] += time() - t
        
        dK_part.shape = (N1, N2*N1)
        
        t = time()
        rhs = np.dot(dK_part, sig)
        t_f[1] += time() - t
        
        sig_slice[-2] = n2
        # so .. image[...] is already S' = [-(K+)(Ko)(K+)(S^T)]^T
        # just need to add [(K+)(dKp)(S'^T)]^T
        t = time()
        k_pln[sig_slice] += np.dot(kpi[n2%2], rhs).transpose()
        t_f[2] += time() - t

    k_pln *= a0_xterms_inv
    print "kernel function creation: %1.2f"%(t_kern,)
    print "kernel inversion: %1.2f, application: %1.2f"%tuple(t_pi.tolist())
    print "forward op creation: %1.2f, application 1: %1.2f, application 2: %1.2f"%tuple(t_f.tolist())
    print "total time: %2.2f"%(t_kern + t_pi.sum() + t_f.sum(),)
    return k_pln

def eval_deghost_full(epi, grad, coefs, chan, vol, r3,
                      l, constraints, mask):
    """Evaluate the ghost correction at a coef vector.
    """
    if not constraints(coefs):
        print 'out of bounds', coefs
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

    util.ifft2(k_pln, inplace=True, shift=True)
    g_nrg = ((np.abs(k_pln)*ghost_mask)**2).sum()
    print 'f(',coefs,') =',g_nrg
    return g_nrg

def eval1D_full_wrap(x, idx, *args):
    # args[2] is a list of 5 coefs.. here just change args[2][0] to x
    args[2][idx] = x
    return eval_deghost_full(*args)

def evalND_full_wrap(x, idc, *args):
    for i, idx in enumerate(idc):
        args[2][idx] = x[i]
    return eval_deghost_full(*args)


def search_axis_full(epi, seeds, axis, l=1.0, chan=0, vol=0, r3=0, mask=None):
    """Do a line search on a given axis of the coeff vector
    """
    grad = get_grad(epi)

    coefs = list(seeds)
    # non-negative constraints
    constraints = [lambda x: x>=0]*len(coefs)
    seed = seeds[axis]
                             
    r = optimize.brent(eval1D_full_wrap, args=(axis, epi, grad, coefs, chan,
                                               vol, r3, l,
                                               constraints, mask),
                       brack = (0, 2*seed or 1),
                       maxiter=50,
                       full_output = 1
                       )
    print r
    return r
