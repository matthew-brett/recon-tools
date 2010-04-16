from recon.operations import Operation, Parameter, ChannelAwareOperation

from recon.operations.GeometricUndistortionK import regularized_inverse

import numpy as np

TIMESTEP = 1.0 # gradient waveform/k-space sample spacing in microseconds

def gtranslate(fx, t):
    global TIMESTEP
    t = np.asarray(t)
    if not t.shape:
        t.shape = (1,)
    len = fx.shape[0]
    lo = (t/TIMESTEP).astype('i')
    hi = lo+1
##     if(lo<0):
##         return fx[0]
##     if(hi>=len):
##         return fx[len-1]
    chi = t/TIMESTEP - lo
    clo = hi - t/TIMESTEP
    fi = (chi*fx[hi] + clo*fx[lo])
    np.putmask(fi, lo<0, fx[0])
    np.putmask(fi, hi>=len, fx[-1])
    return fi

def regularized_solve(A,b,l):
    A2 = np.dot(A.transpose().conjugate(),A)
    n = A2.shape[0]
    l = l*l
    A2.flat[0:n*n:n+1] += l
    b2 = np.dot(A.transpose().conjugate(), b)
    return np.linalg.solve(A2,b2)

def half_solve(A,b):
    return np.dot(A.transpose().conjugate(), b)

def kernel(image, grad, coefs):
    sig1, sig01, sig11, sig21, sig31 = coefs    
    k_adj = 0
    N2 = image.n_pe
    N1 = image.N1
    k = np.zeros((N2,N1,N1), 'D')
    
    Tf = float(image.T_flat); Tr = float(image.T_ramp); T0 = float(image.T0)
    Tl = 2*Tr + Tf
    delT = (Tl-2*T0)/N1
    #delT = image.dwell_time/1e3
    Lx = image.fov_x
    
    a1 = grad.gmaG0*Lx*sig11/(2*np.pi)

    tn = np.arange(N1)*delT + T0
    n1p = np.arange(-N1/2, N1/2)
    for n2 in range(N2):
        k1t = gtranslate(grad.kx, tn-sig1) * grad.gmaG0/(2*np.pi)
        g1t = gtranslate(grad.gx, tn-sig1)
        k[n2] = np.sinc(n1p - (Lx*k1t[:,None]-k_adj) + a1*g1t)        
        tn += Tl
        #k_adj = k_adj ^ 1
    return k

def kernels(image, grad, r3, coefs):
    sig1, sig01, sig11, sig21, sig31 = coefs
    k_adj = 0
    N2 = image.n_pe
    N1 = image.N1
    ko = np.zeros((N2,N1,N1), 'F')
    kp = np.zeros((N2,N2,N1), 'F')
    
    Tf = float(image.T_flat); Tr = float(image.T_ramp); T0 = float(image.T0)
    Tl = 2*Tr + Tf
    delT = (Tl-2*T0)/N1
    #delT = image.dwell_time/1e3
    Lx = image.fov_x; Ly = image.fov_y
    
    a0 = grad.gmaG0*(sig01 + r3*sig31)/(2*np.pi)
    a1 = grad.gmaG0*Lx*sig11/(2*np.pi)
    a2 = grad.gmaG0*Ly*sig21/(2*np.pi)


    tn = np.arange(N1)*delT + T0
    n1p = np.arange(-N1/2, N1/2)
    for n2 in range(N2):
        k1t = gtranslate(grad.kx, tn-sig1) * grad.gmaG0/(2*np.pi)
        g1t = gtranslate(grad.gx, tn-sig1)
        ko[n2] = np.sinc(n1p - (Lx*k1t[:,None]-k_adj) + a1*g1t)
        # multiply a0 term to diagonal of n2'th matrix
        ko.flat[n2*(N1*N1):(n2+1)*(N1*N1):(N1+1)] *= np.exp(1j*a0*g1t)
        for n2p in range(N2):
            kp[n2,n2p] = np.sinc(n2p - n2 + a2*g1t)
        
        tn += Tl
        #k_adj = k_adj ^ 1
    return ko, kp

class UBPC_siemens_1shot (Operation):
    params = (
        Parameter(name='coefs', type='tuple', default=(0,0,0,0,0)),
        Parameter(name='l', type='float', default=1.0),
        )

    @ChannelAwareOperation
    def run(self, image):
        if image.N1 == image.n_pe and image.fov_x > image.fov_y:
            image.fov_y *= 2
            image.jsize = image.isize
        elif image.fov_y == image.fov_x and image.n_pe > image.N1:
            image.fov_y *= (image.n_pe/image.N1)
            image.jsize = image.isize
        print image.fov_y
        grad = Gradient(image.T_ramp, image.T_flat, image.T0,
                        image.n_pe, image.N1, image.fov_x)
        k = kernel(image, grad, self.coefs)
        for n2 in range(image.n_pe):
            sig = image.cdata[:,:,:,n2,:]
            sigshape = sig.shape
            srows = np.product(sigshape[:-1])
            sig.shape = (srows, image.N1)
            
            scorr = regularized_solve(k[n2], sig.transpose(),
                                      self.l).transpose()
##             scorr = half_solve(k[n2], sig.transpose()).transpose()
            scorr.shape = sigshape
            image.cdata[:,:,:,n2,:] = scorr
        image.use_membuffer(0)
        

class Gradient:

    def __init__(self, Tr, Tf, T0, n_echo, N1, Lx):
        self.gx = self.create_grad(Tr, Tf, n_echo)
        self.kx = self.integrate_grad()
        self.kx -= (Tr + Tf)/2.
        As = (Tf + Tr - T0**2/Tr)/2.0
        self.gmaG0 = np.pi*N1/(As*Lx)

    def create_grad(self, Tr, Tf, n_echo):
        global TIMESTEP
        npts_r = Tr/TIMESTEP
        npts_f = Tf/TIMESTEP
        lobe_pts = 2*npts_r + npts_f
        if npts_r:
            ramp_rate = 1.0/npts_r
        glen = lobe_pts*n_echo
        gx = np.zeros(glen)
        polarity = 1.0
        for lobe in range(n_echo):
            i0 = lobe*lobe_pts
            gx[i0:i0+npts_r] = polarity*ramp_rate*np.arange(npts_r)
            i0 += npts_r
            gx[i0:i0+npts_f] = polarity
            i0 += npts_f
            gx[i0:i0+npts_r] = polarity*ramp_rate*(lobe_pts - \
                                                   np.arange(npts_f+npts_r,
                                                             lobe_pts))
            polarity *= -1
        return gx

    def integrate_grad(self):
        kx = np.zeros_like(self.gx)
        global TIMESTEP
        for i in range(1,kx.shape[0]):
            kx[i] = TIMESTEP*(self.gx[i] + self.gx[i-1])/2.0 + kx[i-1]
        return kx

def kx_analytical(Tr, Tf, T0, tn):
    k0 = -(Tr+Tf)/2.
    #tn = np.arange(2*Tr+Tf)
    r1 = tn<Tr
    r2 = (tn>=Tr) & (tn<(Tf+Tr))
    r3 = tn>=(Tf+Tr)
    kx = np.zeros_like(tn)
    kx[r1] = tn[r1]**2/(2*Tr) + k0
    kx[r2] = Tr/2. - Tr + tn[r2] + k0
    kx[r3] = (Tf+Tr) - (2*Tr+Tf - tn[r3])**2/(2*Tr) + k0
    return kx
    

from recon.fftmod import fft1, ifft1
import pylab as P
def upsample(a, m=2):
    L = a.shape[0]
    dt = a.dtype.char
    b = np.zeros((L*m,), dtype=dt.upper())
    b[::m] = m*a
    fft1(b, shift=True, inplace=True)
    b[:(L*m-L)/2] = 0.
    b[(L*m+L)/2:] = 0.
    P.plot(np.abs(b))
    P.show()
    ifft1(b, shift=True, inplace=True)
    return b.astype(dt)
    
from recon import util
def interp_MN(epi, epi_sm, M, N):
    # make a sinc kernel with N rows and M columns
    mt = np.linspace(0, M, M, endpoint=False)
    nt = np.linspace(0, M, N, endpoint=False)
    snc = np.sinc( mt - nt[:,None] ).transpose()
    sm_cdata_shape = list(epi_sm.cdata.shape)
    sm_cdata_shape[-1] = N
    del epi_sm.cdata
    del epi_sm.data
    epi_sm.cdata = util.TempMemmapArray(tuple(sm_cdata_shape), np.complex64)
    for c in range(epi.n_chan):
        epi.load_chan(c)
        epi_sm.load_chan(c)
        epi_sm[:] = np.dot(epi[:], snc)
    epi_sm.use_membuffer(0)
    epi.use_membuffer(0)
    epi_sm.N1 = N
    epi_sm.fov_y = 480
    epi_sm.jsize = epi_sm.isize
    
