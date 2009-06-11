import numpy as np
import pylab as P
from scipy.optimize import fminbound
from scipy import linalg
from recon import util
import time

class NegSvalError(Exception): pass

def simple_kernel(grad, ts, sig1, a0):
    N1 = len(ts)
    dk = 1.0/(N1*grad.dx)
    n1ax = np.arange(-N1/2, N1/2)
    kxt = grad.kxt(ts - sig1) * grad.gmaG0/(2*np.pi)
    g1t = grad.gxt(ts - sig1)
    a0_xterms = np.exp(1j*a0*g1t)
    snc_op = np.dot(np.diag(a0_xterms), np.sinc(kxt[:,None]/dk - n1ax[None,:]))
    
    return snc_op

def diff_op(P, n=1, dt='d'):
    L = np.zeros((P-n, P), dtype=dt)
    pulse = np.zeros((2*n+1), 'd')
    pulse[n] = 1.
    h = np.diff(pulse, n=n)
    lh = len(h)
    for i in xrange(P-n):
        L[i,i:i+lh].real = h
    return L

def svd_or_else(A):
    u, s, vt = None, None, None
    try:
        u, s, vt = linalg.svd(A, compute_uv=True)
        if (s<0).any():
            raise NegSvalError
    except:
        try:
            u, s, vt = linalg.svd(A, compute_uv=True, fast_svd=False)
            if (s<0).any():
                raise NegSvalerror
        except:
            raise Exception("No SVD!")
    return u, s, vt
    
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
    svd_negs = []
    stable_svd_fail = []
    stable_svd_negs = []
    tols = np.zeros(2)
    search_time = 0.
    for n2 in [0,1]:
        #print 'trying',n2
        k1t = grad.kxt(tn0 + n2*Tl - sig1) * grad.gmaG0/(2*np.pi)
        n2_sl = slice(n2,N2,2)
        aref = np.abs(np.trapz(ksp_pln[n2_sl].sum(axis=0), x=k1t))
        snc_op = np.sinc(k1t[:,None]/dk - n1ax[None,:])
        try:
            [u, s, vt] = linalg.svd(snc_op)
            if (s<0).any():
                raise NegSvalError("neg svals")
        except linalg.LinAlgError:
            svd_fail.append(n2)
            s = None
        except NegSvalError:
            svd_negs.append(n2)
            s = None
            #continue
        finally:
            if s is None:
                try:
                    [u, s, vt] = linalg.svd(snc_op, fast_svd=False)
                    if (s<0).any():
                        raise NegSvalError("neg stable svals")
                except np.linalg.LinAlgError:
                    stable_svd_fail.append(n2)
                except NegSvalError:
                    stable_svd_negs.append(n2)
        if s is None:
            continue
        t = time.time()
        tols[n2] = fminbound(eval_area, 1e-10, 1e-1, xtol=1e-5,
                             args=(ksp_pln[n2_sl], u, s, vt, aref))
        search_time += (time.time() - t)
        #print reduce(lambda x,y: x+y, ['-']*40)
    if svd_fail:
        print 'svd failed on:', svd_fail
    if svd_negs:
        print 'neg svals on:', svd_negs
    if stable_svd_fail:
        print 'stable svd failed on:', stable_svd_fail
    if stable_svd_negs:
        print 'neg stable svals on:', stable_svd_negs
    #print "search time:", search_time
    return tols

def inv_op_lcurve_corr(epi, grad, sig1, a0, chan=0, vol=0, sl=0,
                       down_samp=True, diff_order=0):
    N2, N1 = epi.shape[-2:]
    Tl = 2*grad.Tr + grad.Tf
    delT = (Tl-2*grad.T0)/(N1-1)
    ts = np.arange(N1)*delT + grad.T0
    b = epi.cdata[chan,vol,sl]
    L = diff_op(N1, n=diff_order, dt=b.dtype) if diff_order else None
    xreg = np.zeros_like(b)
    for n2 in [0,1]:
        op = simple_kernel(grad, ts + n2*Tl, sig1, a0)
        if diff_order:
            [u,u2,vt,c,s] = linalg.gsvd_matlab(op, L)
            alpha = c.real.diagonal()[diff_order:]
            beta = s[:,diff_order:].real.diagonal()
            s = np.array([alpha, beta])
        else:
            [u,s,vt] = svd_or_else(op)
        
        for xrow, brow in zip(xreg[n2::2], b[n2::2]):
            norm_est = np.dot(brow, brow.conj()).real ** .5
            xrow[:],_ = util.regularized_solve_lcurve(op,brow,L=L,
                                                      u=u,s=s,vt=vt,
                                                      max_lx_norm=5*norm_est)
    if down_samp:
        return xreg[:,::2]
    else:
        return xreg
    

def inv_op_svdreg_corr(epi, grad, sig1, a0, chan=0, vol=0, sl=0,
                       down_samp=True, plotting=False):
    
    N2, N1 = epi.shape[-2:]
    N1_out = N1/2 if down_samp else N1
    n1ax = np.arange(-N1/2,N1/2)
    dk = 1.0/(N1*grad.dx)
    Tl = 2*grad.Tr+grad.Tf
    delT = (Tl-2*grad.T0)/(N1-1)
    tsamps = np.arange(N1)*delT + grad.T0    
    tols = find_tols(epi.cdata[chan,vol,sl], grad, tsamps, Tl, sig1)
    ksp_pln = np.zeros((N2,N1_out), epi.cdata.dtype)
    #for n2 in xrange(N2):
    for n2 in [0,1]:
        k1t = grad.kxt(tsamps + n2*Tl - sig1) * grad.gmaG0/(2*np.pi)
        g1t = grad.gxt(tsamps + n2*Tl - sig1)
        snc_op = np.sinc(k1t[:,None]/dk - n1ax[None,:])
        n2_sl = slice(n2,N2,2)
        try:
            [u,s,vt] = linalg.svd(snc_op)
            if (s<0).any():
                raise linalg.LinAlgError("negative svals")
        except linalg.LinAlgError, err:
            print "no svd for",(chan,sl,n2),"sig1=",sig1
            print err.args
            try:
                [u,s,vt] = linalg.svd(snc_op, fast_svd=False)
                if (s<0).any():
                    raise NegSvalError
            except linalg.LinAlgError:
                print "stable svd failed for",(chan,sl,n2),"sig1=",sig1
                raise Exception
            except NegSvalError:
                print 'stable svd gave neg svals for',(chan,sl,n2),'sig1=',sig1
                raise Exception
        si = np.where(s < tols[n2], 0, 1/s)
        #si = np.where(s < 1e-2, 0, 1/s)
        Hpinv = np.dot(vt.conjugate().transpose(),
                       si[:,None]*u.conjugate().transpose())

        a0_xterm_inv = np.exp(-1j*a0*g1t)
        partcorr_echos = epi.cdata[chan,vol,sl,n2_sl]*a0_xterm_inv
        if down_samp:
            Hpinv = Hpinv[::2]
        ksp_pln[n2_sl] = np.dot(partcorr_echos, Hpinv.T)
    if plotting:
        P.figure()
        P.imshow(np.abs(util.ifft2(ksp_pln)))
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
                                      epi.n_slice, epi.n_pe, epi.N1/2),
                                     epi.cdata.dtype)
    else:
        cdata = epi.cdata
    for c in range(epi.n_chan):
        for s in range(epi.n_slice):
            ksp_pln = forward_op_corr(epi, grad, cf[c,s,0], cf[c,s,3],
                                      chan=c, vol=0, sl=s, down_samp=down_samp)
            cdata[c,0,s] = ksp_pln

    if down_samp:
        del epi.cdata
        epi.cdata = cdata
        epi.N1 /= 2
    epi.use_membuffer(0)

def deghost_epi_invops_svdreg(epi, grad, cf, down_samp=True):
    if down_samp:
        cdata = util.TempMemmapArray((epi.n_chan, epi.n_vol,
                                      epi.n_slice, epi.n_pe, epi.N1/2),
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

def deghost_epi_invops_lcurve(epi, grad, cf, diff_order=0, down_samp=True):
    if down_samp:
        cdata = util.TempMemmapArray((epi.n_chan, epi.n_vol,
                                      epi.n_slice, epi.n_pe, epi.N1/2),
                                     epi.cdata.dtype)
    else:
        cdata = epi.cdata
    for c in range(epi.n_chan):
        for s in range(epi.n_slice):
            ksp_pln = inv_op_lcurve_corr(epi, grad, cf[c,s,0], cf[c,s,3],
                                         chan=c, vol=0, sl=s,
                                         diff_order=diff_order,
                                         down_samp=down_samp)
            cdata[c,0,s] = ksp_pln
    if down_samp:
        del epi.cdata
        epi.cdata = cdata
        epi.N1 /= 2
    epi.use_membuffer(0)
    
            
