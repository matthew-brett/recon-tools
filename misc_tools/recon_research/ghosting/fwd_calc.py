import numpy as np
import scipy as sp
import scipy.integrate
import pylab as P
from recon import util

def eyeball_fmap(shape, mu, fwhm, max_freq=1000., over_samp=1):
    """ Create a two-peaked field map with the superposition of two
    Gaussians with means at mu[0] and mu[1]

    Parameter
    _________
    shape : 3-tuple
        containing dim sizes
    mu : len-2 list
        containing index coordinates of peak positions in (y,x) order
    fwhm : float [or list]
        width of distribution in all [or separate in z,y,x order] directions
    max_freq: float
        maximum value at peaks
    over_samp: int
        boost resolution in the y-direction
    """
    new_shape = [l for l in shape]
    new_shape[-2] *= over_samp
    fwhm_scale = 2*((2*np.log(2))**0.5)
    try:
        sz, sy, sx = map(lambda x: x/fwhm_scale, fwhm)
    except TypeError:
        sz = sy = sx = fwhm / fwhm_scale
    sy *= over_samp

    nz, ny, nx = np.indices(new_shape)

    pk1y, pk1x = mu[0]; pk1y *= over_samp
    g1 = np.exp(-1*( (nx-pk1x)**2/(2*sx**2) + (ny-pk1y)**2/(2*sy**2) ))
    g1 *= max_freq

    pk2y, pk2x = mu[1]; pk2y *= over_samp
    g2 = np.exp(-1*( (nx-pk2x)**2/(2*sx**2) + (ny-pk2y)**2/(2*sy**2) ))
    g2 *= max_freq

    # the z-dimension distribution should peak around slice 3??
    # with fwhm = 4??
    g3 = np.exp(-1*( (nz-11)**2/(2*sz**2) ))

    return (g1+g2) *g3

def expanding_fmap(shape, max_freq=1000., over_samp=1,
                   boost_region=None, boost_strength=1.5,
                   boost_direction='pos'):
    Q2 = shape[-2]
    new_shape = [dim for dim in shape]
    new_shape[-2] *= over_samp
    fmap = np.empty(tuple(new_shape), 'd')
    p = np.polyfit(np.array([-Q2/2., -Q2/4., 0., Q2/4., Q2/2.]),
                   np.array([0., max_freq, 0., -max_freq, 0.]), 3)
    fp = np.polyval(p, np.linspace(-Q2/2., Q2/2., Q2*over_samp, endpoint=False))
    fmap[:] = fp[None,:,None]
    if boost_region:
        b_slices, b_rows, b_cols = boost_region
        assert type(b_slices)==type(b_rows)==type(b_cols)==type(slice(None))
        rstart = over_samp*b_rows.start; rstop = over_samp*(b_rows.stop+1)-1
        bwid = int(round((rstop-rstart+1)*1.1))
        cnt = (rstart+rstop)/2
        if bwid%2:
            new_brows = slice(max(0, cnt-bwid/2),min(Q2*over_samp,cnt+bwid/2+1))
        else:
            new_brows = slice(max(0, cnt-bwid/2),min(Q2*over_samp,cnt+bwid/2))
        boost = np.ones(over_samp*Q2)
        boost[new_brows] += (boost_strength-1.)*np.hanning(bwid)
        b_region = (b_slices, slice(None), b_cols)
        fmap[b_region] *= boost[None,:,None]
    return fmap

def contracting_fmap(shape, **kws):
    return -expanding_fmap(shape, **kws)

def patch_fmap(fmap, patch_region, patch_blending=5, order=5):
    dims = fmap.shape
    patch_limits = [((0 if not sl.start else sl.start),
                     (d if not sl.stop else sl.stop)) \
                    for sl,d in zip(patch_region, dims)]
    for sl in range(fmap.shape[-3]):
    #for sl in [13]:
        def valid_pt(x,y):
            return x >= patch_limits[-1][0] and x < patch_limits[-1][1] \
                   and y >= patch_limits[-2][0] and y < patch_limits[-2][1]
        nz = (fmap[sl].T).nonzero()
        nz_x, nz_y = nz
        dnz_y = np.diff(nz_y)
        putative_holes = (dnz_y!=1).nonzero()[0]
        holes = []
        for h in putative_holes:
            if h == len(nz_x)-1: break
            if nz_x[h] == nz_x[h+1] and valid_pt(nz_x[h], nz_y[h]):
                # this is an actual hole
                holes.append( (sl, slice(nz_y[h]+1, nz_y[h+1]), nz_x[h]) )
        for h in holes:
            sl, col = h[0], h[2]
            x_end = fmap[sl,h[1].stop:,col].nonzero()[0] + h[1].stop
            x_start = np.arange(h[1].start-len(x_end),h[1].start)
            x = np.concatenate((x_start, x_end))
            y = fmap[sl,x,col]
            p = np.polyfit(x, y, order)#, rcond=0.005)
            x_interp = np.arange(x[0], x[-1]+1)
            y_interp = np.polyval(p, x_interp)
            blend_length = max(patch_blending, int(2*round(len(x_start)/3.)))

            x_patch = np.arange(x_start[-1]-blend_length,
                                x_end[0]+blend_length+1)
##             x_patch = x_interp
            y_patch = np.polyval(p, x_patch)
            fmap[sl,x_patch,col] = y_patch
            
        
# sig1: 0.72864776906742923
# sig01: 0.050122199999999999
# sig11: 0.11355542311841092
# sig21: 0.0011661606117854752
# sig31: 8.6428803400000003e-05

# Sp = KS, where S[j] <-- S[n2,n1], and K[i,j] <-- K[n2,n1,n2',n1']

# K[n2,n1,n2',n1'] =
# sum<q2,q1>{ exp(i2PIq1n1'/N1) * exp(-i2PIq1[Lx*k(tn-sig1)-a1*g(tn-sig1)]/N1) *
#             exp(i2PIq2n2'/N2) * exp(-i2PIq2[n2 - a2*g(tn-sig1)]/N2) *
#             exp(i*(tn-sig1)*fmap[q2,q1])
#
# which is IFFT2D< q=>n' >{f1(n2,n1,q1)f2(n2,n1,q2)*fm(n2,n1,q2,q1)}

def perturb_image(img, grad, sig1, sig01, sig11, sig21, sig31, fmap, **kw):
    Q3, N2, N1 = img.shape[-3:]
    q1_ax = n1_ax = np.arange(-N1/2, N1/2)
    q2_ax = n2_ax = np.arange(-N2/2, N2/2)
    Lx = img.fov_x; Ly = img.fov_y
    #Tr = img.T_ramp; Tf = img.T_flat; T0 = img.T0
    Tr = kw.get('Tr', 150.)
    Tf = kw.get('Tf', 120.)
    T0 = kw.get('T0', 31.)
    Tl = 2*Tr+Tf
    delT = (Tl - 2*T0)/float(N1-1)
    gmaG0bar = grad.gmaG0/(2*np.pi)
    a1 = Lx*gmaG0bar*sig11
    a2 = Ly*gmaG0bar*sig21
    Sp_all = np.empty((img.n_chan*img.n_vol, Q3, N2, N1), img.cdata.dtype)
    for s in xrange(Q3):
    #for s in [0,1,2]:
        S = img.cdata[:,:,s,:,:].transpose(2,3,0,1).copy()
        S.shape = (N2*N1, img.n_chan*img.n_vol)
        Sp = np.empty((N2, N1, img.n_chan*img.n_vol), S.dtype)
        a0 = (sig01+sig31*s*-img.ksize)
        a0_xterms = np.empty((N2,N1), S.dtype)
        tn = np.arange(0.0, N1)*delT + T0
        for n2 in xrange(N2):
            print (s,n2)
            kxt = grad.kxt(tn-sig1)
            gxt = grad.gxt(tn-sig1)
            a0_xterms[n2] = np.exp(1j*a0*gxt)
            f1_arg = (Lx*gmaG0bar*kxt-a1*gxt)[:,None] * q1_ax[None,:]
            f1 = np.exp(-2j*np.pi*f1_arg/N1)
            f2_arg = (n2 - N2/2. - a2*gxt)[:,None] * q2_ax[None,:]
            f2 = np.exp(-2j*np.pi*f2_arg/N2)
            # fmap is measured in rad/s, tn is in microsec, so scale by 1e-6
            fm = np.exp(1j*(tn-sig1)[:,None,None]*fmap[s,:,:]*1e-6)
            fnq = f2[:,:,None] * f1[:,None,:] * fm[:,:,:]
            util.ifft2(fnq, inplace=True, shift=True)
            fnq.shape = (N1,N2*N1)
            Sp[n2] = np.dot(fnq, S)
            tn += Tl

        Sp *= a0_xterms[:,:,None]
        Sp_all[:,s,:,:] = Sp.transpose(2,0,1).copy()

    Sp_all.shape = (img.n_chan, img.n_vol, Q3, N2, N1)
    img.cdata[:] = Sp_all
    img.use_membuffer(0)

def perturb_image_noghosts_constFE(img, fmap, **kw):
    """ Applies B0 inhomogeneity phase distortion to the array img

    Parameters
    __________
    img : ndarray
        a 5-dimensional array representing the n-channel image
    fmap : ndarray
        a 3-dimensional array representing the B0 field inhomo map
    accel : int/float
        controls the slope of the time function with respect to n2
    n2_freqs : ndarray, optional
        frequencies at which to calculate the DFT (useful for distorting ACS)
    rev_pe : Boolean, optional
        reverses the timing function (simulates reverse phase encoding)
    Tl : float, optional
        echo spacing (in microsecs)
    pe_offset : int, optional
        number of echo spacings to play out before distorting rows

    Returns
    _______
    None

    """
    if not issubclass(type(img), np.ndarray):
        img = img.cdata[:]
    n_chan, n_vol, Q3, N2, N1 = img.shape
    Q2 = fmap.shape[-2]
    q1_ax = n1_ax = np.arange(-N1/2, N1/2)
    q2_ax = np.arange(-Q2/2, Q2/2)
    accel = kw.get('accel', 1)
    n2_ax = kw.get('n2_freqs', np.arange(-N2/2, N2/2))
    rev_pe = kw.get('rev_pe', False)
    Tl = kw.get('Tl', 500.)
    pe_offset = kw.get('pe_offset', 0)
    #acs_mode = kw.get('acs_mode', False)
    if 'n2_freqs' not in kw and N2==Q2:
        idft_func = lambda x: util.ifft1(x, shift=True)
    else:
        def idft_func(arr):
            util.ifft1(arr, shift=True, inplace=True)
            # n2 in {-N2/2,...,N2/2} corresponds to indices of
            # n2i in {-N2/2+Q2/2, ..., N2/2+Q2/2}
            n2_idx = (n2_ax + Q2/2).astype('i')
            arr_xf = arr[...,n2_idx].copy()
            return arr_xf
##     if not acs_offset and accel > 1:
##         # in accelerated case, the first measured line is always 1,
##         # so force t_n2[1] = 0
##         t_n2 = np.arange(-1,N2-1)/float(accel) * Tl * 1e-6
##         #t_n2 = np.arange(-1,N2-1)/int(accel) * Tl * 1e-6
##         #t_n2 = np.arange(-1,N2-1)/float(accel) * Tl * 1e-6
##     else:
##         t_n2 = (acs_offset + np.arange(N2)/float(accel)) * Tl * 1e-6
##         #t_n2 = np.arange(N2)/int(accel) * Tl * 1e-6
##         #t_n2 = np.arange(N2)/float(accel) * Tl * 1e-6
    t_n2 = (pe_offset + np.arange(N2)/float(accel)) * Tl * 1e-6
    if rev_pe:
        t_n2 = t_n2[::-1]
    print t_n2
    util.ifft1(img, inplace=True, shift=True)
    for s in xrange(Q3):
        print "sl:",s
        fm_s = fmap[s,:,:].transpose().copy()
        fm = np.exp(1j*(t_n2[None,:,None]*fm_s[:,None,:]))
        fm *= np.exp(-1j*2*np.pi*n2_ax[:,None]*q2_ax[None,:]/Q2)
        # fm is (q1,n2,q2) --> (q1,n2,n2')
        fm = idft_func(fm)
        for q1 in xrange(N1):
            S = img[:,:,s,:,q1].transpose(2,0,1).copy()
            S.shape = (N2,n_chan*n_vol)
            Sp = np.dot(fm[q1], S)
            Sp.shape = (N2, n_chan, n_vol)
            img[:,:,s,:,q1] = Sp.transpose(1,2,0)
    util.fft1(img, inplace=True, shift=True)

class CachedFunc:

    def __init__(self, func):
        self.func = func
    def _cached_func_call(self, args):
        if not hasattr(self, '_fvals'): self._fvals = {}
        if self._fvals.has_key(args):
            #print 'return from cache'
            return self._fvals[args]
        else:
            #print 'new func call'
            a = self.func(*args)
            self._fvals[args] = a
            return a

    def __call__(self, *args):
        return self._cached_func_call(args)

    def _reset(self):
        try:
            del self._fvals
        except:
            pass
        self._fvals = {}

@CachedFunc
def integrand(omega, fm_r, n2p, t_n2, n2):
    return np.exp(1j*fm_r*t_n2)*np.exp(-1j*n2*omega)*np.exp(1j*omega*n2p)
    

def int_func(omega, n2p, fm, tn, n2, fov):
    # omega is in [-PI, PI]
    r = fov*omega/(2*np.pi)
    #is len(tn) == N2' ? 
    dr = fov/tn.shape[0]
    # see if fm(r) is callable)
    try:
        fm_r = fm(r/dr)
    except:
        fm_r = util.feval_bilinear(fm, r, dr)
    try:
        fm_r = fm_r[0]
    except:
        pass
    t_n2 = tn[n2]
    n2 -= tn.shape[0]/2
    # these args must be hashable (ie: no ndarrays)
    return integrand(omega, fm_r, n2p, t_n2, n2)

def integrand_r(*args):
    return int_func(*args).real

def integrand_i(*args):
    return int_func(*args).imag
    

def idtft(f_omega, n2_freqs, f_omega_args):
    """integrates f_omega from -PI to PI for each of n2_freqs..
    integrand is computed by f_omega(omega, *(n2_freq+f_omega_args))
    """
    fn = np.zeros(n2_freqs.shape, 'D')
    
    for i, n2 in enumerate(n2_freqs):
        try:
            f_iter = iter(f_omega)
            cplx_itegrand = True
        except:
            f_omega = (f_omega,)
            f_iter = iter(f_omega)
        z = np.zeros((2,), 'd')
        for zi,f in zip((0,1), f_iter):
            z[zi] = sp.integrate.quad(f, -np.pi, np.pi,
                                     args=((n2,)+f_omega_args))[0]

        fn[i] = (z[0] + 1j*z[1])/(2*np.pi)
    return fn


def compare_dfts(fm, tn, n2range, do_idft=True):
    n2freqs = np.arange(-128,128.)
    n2_ax_1x = np.arange(-32,32)
    n2_ax_4x = np.arange(-128,128)
    for n2 in n2range:
        rn1 = np.linspace(-32.,32.,64,endpoint=False)
        rn2 = np.linspace(-32.,32.,256,endpoint=False)
        f_1x = util.ifft1(np.exp(1j*tn[n2]*fm(rn1)) * \
                          np.exp(-2j*np.pi*rn1*(n2-32)/64.))
        f_4x = util.ifft1(np.exp(1j*tn[n2]*fm(rn2)) * \
                          np.exp(-2j*np.pi*rn2*(n2-32)/64.))
        if do_idft:
            idft_fn = idtft((integrand_r, integrand_i), n2freqs,
                            (fm, tn, n2, 240.))
            P.plot(n2_ax_4x, np.abs(idft_fn))
        P.plot(n2_ax_1x, np.abs(f_1x))
        P.plot(n2_ax_4x, np.abs(f_4x))
        P.plot(n2_ax_1x, np.abs(f_4x[128-32:128+32]), 'k--')
        P.show()
        integrand._reset()
        if not n2%4:
            chk_continue = raw_input('??? ')
            if chk_continue.lower()[0] != 'y':
                break
    
def basic_fix_up(img, grad, sig1, Tr, Tf, T0):
    N2, N1 = img.shape[-2:]
    # time-reverse neg-trajectory rows
    rev = img.cdata[:,:,:,1::2,:].copy()
    img.cdata[:,:,:,1::2,:] = rev[...,::-1]
    del rev

    # make sinc interpolator to regrid to uniform k-space
    As = (Tf + Tr - T0**2/Tr)
    delT = (2*Tr + Tf - 2*T0)/float(N1-1)
    n_ramp = int( N1 * (Tr**2 - T0**2)/(2*Tr*As) )
    n_flat = min(int(N1)-1,
                 int( N1*(2*Tr*Tf + Tr**2 - T0**2)/(2*Tr*As) ))
    tn = np.zeros(N1)
    r1 = np.arange(0,n_ramp+1)
    tn[r1] = np.power(T0**2 +2*r1*Tr*As/(N1-1), 0.5)
    r2 = np.arange(n_ramp+1,n_flat+1)
    tn[r2] = r2*As/(N1-1) + Tr/2. + T0**2/(2*Tr)
    r3 = np.arange(n_flat+1,N1)
    tn[r3] = (Tf+2*Tr) - np.power(2*Tr*(Tf+Tr-T0**2/(2*Tr)-r3*As/(N1-1)), 0.5)

    _t = (tn-T0)/delT
    s_idx = _t[None,:] - np.arange(N1)[:,None]
    snc_kernel = np.sinc(_t[None,:] - np.arange(N1)[:,None]).astype('F')

    img.cdata[:] = np.dot(img.cdata.reshape(np.product(img.cdata.shape[:-1]),
                                                       img.cdata.shape[-1]),
                          snc_kernel).reshape(img.cdata.shape)

    # make simple phase correction based no-ramp-samp model
    pramp = sig1*grad.gmaG0*img.isize
    k_phs_planar = np.exp(-1j*np.power(-1, np.arange(N2))[:,None] * \
                          (pramp*np.arange(-N1/2.,N1/2.))[None,:])
    util.apply_phase_correction(img.cdata, k_phs_planar)

    img.use_membuffer(0)
