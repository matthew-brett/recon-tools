import numpy as np
from recon.operations import UBPC_siemens_1shot_new as UBPC
from recon import util
from time import time

def kernels_lite(image, grad, r3, n1, coefs):
    # let a0 float
    sig1, sig11, sig21, a0 = coefs
    k_adj = 0
    N2 = image.n_pe
    N1 = image.N1
    ko = np.zeros((N2,N1,N1), 'D')
    dk = np.zeros((N2,N2), 'D')
    #a0_xterms_inv = np.zeros((N2,), 'D')
    
    Tf = float(image.T_flat); Tr = float(image.T_ramp); T0 = float(image.T0)
    Tl = 2*Tr + Tf
    delT = (Tl-2*T0)/(N1-1)
    #delT = image.dwell_time/1e3
    Lx = image.fov_x; Ly = image.fov_y
    
    #a0 = grad.gmaG0*(sig01 + r3*sig31)/(2*np.pi)
    a1 = grad.gmaG0*Lx*sig11/(2*np.pi)
    a2 = grad.gmaG0*Ly*sig21/(2*np.pi)


    tn = np.arange(N1)*delT + T0
    n1p = np.arange(-N1/2, N1/2)
    for n2 in range(N2):
        # on negative echos, make a small adjustment to shift by delk
        # delK is gmaG0 * dt_eff / 2PI (dt_eff is effective rectangular gradient
        # duration divided by N1)
        
        k1t = UBPC.gtranslate(grad.kx, (tn-sig1)) * grad.gmaG0/(2*np.pi)
        g1t = UBPC.gtranslate(grad.gx, (tn-sig1))
        ko[n2] = np.sinc(n1p - (Lx*k1t - a1*g1t)[:,None])
        ## NOOO!
##         # multiply a0 term to diagonal of n2'th matrix
##         ko.flat[n2*(N1*N1):(n2+1)*(N1*N1):(N1+1)] *= np.exp(1j*a0*g1t)
        for n2p in range(N2):
            dk[n2,n2p] = -np.sinc(n2p - n2 + a2*g1t[n1])
        dk[n2,n2] += 1.0
        
        tn += Tl
        k_adj = k_adj ^ 1

    t_fix_n1 = T0 + n1*delT + np.arange(N2)*Tl - sig1
    g1t_n1 = UBPC.gtranslate(grad.gx, t_fix_n1)
    a0_xterms_inv = np.exp(-1j*a0*g1t_n1)
    return ko, dk, a0_xterms_inv
    

def eval_deghost(epi, grad, coefs, chan, vol, r3, l, col, constraints):
    """Evaluate the ghost correction at a coef vector by projecting the
    ghosts onto a single column.
    """
    if not constraints(coefs):
        print 'out of bounds', coefs
        return 1e11
    N2, N1 = epi.shape[-2:]
    if col < 0: col = N1/2
    col_mask = np.zeros(N2)
    col_mask[:N2/4] = 1.0
    col_mask[3*N2/4:] = 1.0
    t = time()
    ko, dk, a0_xterms_inv = kernels_lite(epi, grad, r3, col, coefs)
    t_kern = time() - t
    kpi = ko.copy()
    sig_slice = [slice(None)]*2
    k_pln = epi.cdata[chan, vol, r3].copy()
    t_pi = np.zeros(2)
    for n2 in range(N2):
        sig_slice[-2] = n2
        t = time()
        kpi[n2] = UBPC.regularized_inverse(kpi[n2], l)
        t_pi[0] += time() - t

        t = time()
        sig = k_pln[sig_slice].transpose().copy()
        sig = np.dot(kpi[n2], sig)
        sig = np.dot(ko[n2], sig)
        sig = -np.dot(kpi[n2], sig)
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
    dKfwd = ko[:,col,None,:] * dk[:,:,None]
    dKfwd.shape = (N2, N2*N1)
    t_f[0] += time() - t

    t = time()
    rhs = np.dot(dKfwd, sig)
    t_f[1] += time() - t

    t = time()
    col_vec += np.dot(kpi[:,col,:], rhs)
    col_vec *= a0_xterms_inv
    t_f[2] += time() - t

    print "kernel function creation: %1.2f"%(t_kern,)
    print "kernel inversion: %1.2f, application: %1.2f"%tuple(t_pi.tolist())
    print "forward op creation: %1.2f, application 1: %1.2f, application 2: %1.2f"%tuple(t_f.tolist())
    print "total time: %2.2f"%(t_kern + t_pi.sum() + t_f.sum(),)

##     return col_vec*col_mask
    util.ifft1(col_vec, inplace=True, shift=True)
    g_nrg = ((np.abs(col_vec)*col_mask)**2).sum()
    print 'f(',coefs,') =',g_nrg
    return g_nrg

def kernels(image, grad, r3, coefs):
    sig1, sig11, sig21, a0 = coefs
    k_adj = 0
    N2 = image.n_pe
    N1 = image.N1
    ko = np.zeros((N2,N1,N1), 'D')
    dk = np.zeros((N2,N1,N2), 'D')
    a0_xterms_inv = np.zeros((N2,N1), 'D')
    
    Tf = float(image.T_flat); Tr = float(image.T_ramp); T0 = float(image.T0)
    Tl = 2*Tr + Tf
    delT = (Tl-2*T0)/(N1-1)
    #delT = image.dwell_time/1e3
    Lx = image.fov_x; Ly = image.fov_y
    
    #a0 = grad.gmaG0*(sig01 + r3*sig31)/(2*np.pi)
    a1 = grad.gmaG0*Lx*sig11/(2*np.pi)
    a2 = grad.gmaG0*Ly*sig21/(2*np.pi)


    tn = np.arange(N1)*delT + T0
    n1p = np.arange(-N1/2, N1/2)
    for n2 in range(N2):
        # on negative echos, make a small adjustment to shift by delk
        k1t = UBPC.gtranslate(grad.kx, (tn-sig1)) * grad.gmaG0/(2*np.pi)
        g1t = UBPC.gtranslate(grad.gx, (tn-sig1))
        ko[n2] = np.sinc(n1p - (Lx*k1t - a1*g1t)[:,None])
##         # multiply a0 term to diagonal of n2'th matrix
##         ko.flat[n2*(N1*N1):(n2+1)*(N1*N1):(N1+1)] *= np.exp(1j*a0*g1t)
        a0_xterms_inv[n2] = np.exp(-1j*a0*g1t)
        for n2p in range(N2):
            dk[n2,:,n2p] = -np.sinc(n2p - n2 + a2*g1t)
        dk[n2,:,n2] += 1.0
        
        tn += Tl
        #k_adj = k_adj ^ 1
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
    for n2 in range(N2):
        sig_slice[-2] = n2
        t = time()
        kpi[n2] = UBPC.regularized_inverse(kpi[n2], l)
        t_pi[0] += time() - t

        t = time()
        sig = k_pln[sig_slice].transpose().copy()
        sig = np.dot(kpi[n2], sig)
        sig = np.dot(ko[n2], sig)
        sig = -np.dot(kpi[n2], sig)
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
        dK_part = ko[n2,:,None,:] * dk[n2,:,:,None]
        t_f[0] += time() - t
        
        dK_part.shape = (N1, N2*N1)
        
        t = time()
        rhs = np.dot(dK_part, sig)
        t_f[1] += time() - t
        
        sig_slice[-2] = n2
        # so .. image[...] is already S' = [-(K+)(Ko)(K+)(S^T)]^T
        # just need to add [(K+)(dKp)(S'^T)]^T
        t = time()
        k_pln[sig_slice] += np.dot(kpi[n2], rhs).transpose()
        t_f[2] += time() - t

    k_pln *= a0_xterms_inv
    print "kernel function creation: %1.2f"%(t_kern,)
    print "kernel inversion: %1.2f, application: %1.2f"%tuple(t_pi.tolist())
    print "forward op creation: %1.2f, application 1: %1.2f, application 2: %1.2f"%tuple(t_f.tolist())
    print "total time: %2.2f"%(t_kern + t_pi.sum() + t_f.sum(),)
    return k_pln

