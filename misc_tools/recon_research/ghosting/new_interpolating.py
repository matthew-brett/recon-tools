import numpy as np
import pylab as P
from scipy.optimize import fminbound
from recon import util
import time

def find_tols(ksp_pln, grad, tn0, Tl, sig1):
    N2, N1 = ksp_pln.shape
    dk = 1.0/(N1*grad.dx)
    target_dk = 1/480. #1/240.
    n1ax = np.arange(-N1/2,N1/2)
    def eval_area(x, echo, u, s, vt, aref):
        si = np.where(s < x, 0, 1/s)
        Hpi = np.dot(vt.conjugate().transpose(),
                     si[:,None]*u.conjugate().transpose())
        #echo_fix = np.dot(Hpi, echo)
        echo_fix = np.dot(echo, Hpi.T)
        afix = np.abs(np.trapz(echo_fix.sum(axis=0), dx=target_dk))
        pct = np.abs((afix-aref)/aref)
        #print 'f(',x,') =', pct
        return pct-.001
    
    svd_fail = []
    tols = np.zeros(N2)
    search_time = 0.
    for n2 in [0,1]:
        #print 'trying',n2
        k1t = grad.kxt(tn0 + n2*Tl - sig1) * grad.gmaG0/(2*np.pi)
        n2_sl = slice(n2,N2,2)
        aref = np.abs(np.trapz(ksp_pln[n2_sl].sum(axis=0), x=k1t))
        snc_op = np.sinc(k1t[:,None]/dk - n1ax[None,:])
        try:
            [u, s, vt] = np.linalg.svd(snc_op, 1, 1)
        except:
            svd_fail.append(n2)
            continue
        t = time.time()
        tols[n2] = fminbound(eval_area, 1e-10, 1e-1, xtol=1e-5,
                             args=(ksp_pln[n2_sl], u, s, vt, aref))
        search_time += (time.time() - t)
        #print reduce(lambda x,y: x+y, ['-']*40)
    if svd_fail:
        print 'svd failed on:', svd_fail
    #print "search time:", search_time
    return tols

def inv_op_svdreg_corr(epi, grad, sig1, a0, chan=0, vol=0, sl=0,
                       down_samp=True, plotting=False):
    
    N2, N1 = epi.shape[-2:]
    N1_out = N1/2 if down_samp else N1
    n1ax = np.arange(-N1/2,N1/2)
    dk = 1.0/(N1*grad.dx)
    Tl = 2*grad.Tr+grad.Tf
    delT = (Tl-2*grad.T0)/(N1-1)
    tsamps = np.arange(128)*delT + grad.T0    
    tols = find_tols(epi.cdata[chan,vol,sl], grad, tsamps, Tl, sig1)
    ksp_pln = np.zeros((N2,N1_out), epi.cdata.dtype)
    #for n2 in xrange(N2):
    for n2 in [0,1]:
        k1t = grad.kxt(tsamps + n2*Tl - sig1) * grad.gmaG0/(2*np.pi)
        g1t = grad.gxt(tsamps + n2*Tl - sig1)
        snc_op = np.sinc(k1t[:,None]/dk - n1ax[None,:])
        n2_sl = slice(n2,N2,2)
        try:
            [u,s,vt] = np.linalg.svd(snc_op, 1, 1)
        except:
            print "no svd for",(chan,sl,n2),"sig1=",sig1
            raise Exception
        #si = np.where(s < tols[n2], 0, 1/s)
        si = np.where(s < 1e-2, 0, 1/s)
        Hpinv = np.dot(vt.conjugate().transpose(),
                       si[:,None]*u.conjugate().transpose())

        a0_xterm_inv = np.exp(-1j*a0*g1t)
        partcorr_echos = epi.cdata[chan,vol,sl,n2_sl]*a0_xterm_inv
        if down_samp:
            Hpinv = Hpinv[::2]
        ksp_pln[n2_sl] = np.dot(partcorr_echos, Hpinv.T)
    if plotting:
        P.figure()
        P.imshow(np.abs(util.ifft2(ksp_fix)))
    return ksp_pln

def tn_regrid(T0, Tr, Tf, N1, nr, nf):
    As = (Tf + Tr - T0**2/Tr)
    r1 = np.arange(0, nr+1)
    r2 = np.arange(nr+1, nf+1)
    r3 = np.arange(nf+1, N1)
    t = np.zeros(N1)
    t[r1] = np.power(T0**2 + 2*r1*Tr*As/(N1-1), 0.5)
    t[r2] = r2*As/(N1-1) + Tr/2 + T0**2/(2*Tr)
    t[r3] = (Tf+2*Tr) - np.power(2*Tr*(Tf+Tr-T0**2/(2*Tr) - r3*As/(N1-1)), 0.5)
    return t
    

def forward_op_corr(epi, grad, sig1, a0, chan=0, vol=0, sl=0, down_samp=True):
    N2,N1 = epi.shape[-2:]
    N1_out = N1/2 if down_samp else N1
    Tl = 2*grad.Tr+grad.Tf
    delT = (Tl-2*grad.T0)/(N1-1)
    tn = np.arange(128)*delT + grad.T0
    tn_ngrid = tn_regrid(grad.T0, grad.Tr, grad.Tf, N1, epi.n_ramp, epi.n_flat)
    ksp_pln = np.zeros((N2,N1_out), epi.cdata.dtype)
    def new_grid_times(n2):
        if n2%2:
            return (n2+1)*Tl - tn_ngrid
        else:
            return tn_ngrid + n2*Tl
            
    def sampled_grid_times(n2, sig1):
        return tn + n2*Tl - sig1
    
    for n2 in xrange(N2):
        ngrid = new_grid_times(n2)
        sgrid = sampled_grid_times(n2, sig1)
        g1t = grad.gxt(sgrid)
        op = np.sinc((ngrid[:,None] - sgrid)/delT)
        a0_xterm_inv = np.exp(-1j*a0*g1t)
        partcorr_echo = epi.cdata[chan,vol,sl,n2]*a0_xterm_inv
        if down_samp:
            op = op[::2]
        ksp_pln[n2] = np.dot(op, partcorr_echo)
    return ksp_pln


def deghost_epi_fwdops(epi, grad, cf, down_samp=True):
    if down_samp:
        cdata = util.TempMemmapArray((epi.n_chan, epi.n_vol,
                                      epi.n_sl, epi.n_pe, epi.N1/2),
                                     epi.cdata.dtype)
    else:
        cdata = epi.cdata
    for c in range(epi.n_chan):
        for s in range(epi.n_slice):
            ksp_pln = forward_op_corr(epi, grad, cf[c,s,0], cf[c,s,3],
                                      chan=c, vol=0, sl=s, down_samp=down_samp)
            epi.cdata[c,0,s] = ksp_pln

    if down_samp:
        del epi.cdata
        epi.cdata = cdata
        epi.N1 /= 2
    epi.use_membuffer(0)

def deghost_epi_invops_svdreg(epi, grad, cf, down_samp=True):
    if down_samp:
        cdata = util.TempMemmapArray((epi.n_chan, epi.n_vol,
                                      epi.n_sl, epi.n_pe, epi.N1/2),
                                     epi.cdata.dtype)
    else:
        cdata = epi.cdata
    for c in range(epi.n_chan):
        for s in range(epi.n_slice):
            ksp_pln = inv_op_svdreg_corr(epi, grad, cf[c,s,0], cf[c,s,3],
                                         chan=c, vol=0, sl=s,
                                         down_samp=down_samp)
            cdata[c,0,s] = ksp_pln
    if down_samp:
        del epi.cdata
        epi.cdata = cdata
        epi.N1 /= 2
    epi.use_membuffer(0)
            
