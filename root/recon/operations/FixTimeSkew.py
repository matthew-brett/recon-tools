import numpy as np, pylab as P
from recon.fftmod import fft1 as fft, ifft1 as ifft
from recon.operations import Operation, Parameter, verify_scanner_image, \
     ChannelIndependentOperation
from recon.util import lin_regression
from recon.inplane_xforms import reverse

### for bad artifact see buchsbaum_13Feb07/111_epi_run2.fid/ slice 9

def circularize(a):
    To = a.shape[-1]
    T = 3*To -2 + To%2
    buf_dtype = np.dtype(a.dtype.char.upper())
    ts_buf = np.empty(a.shape[:-1]+ (T,), buf_dtype)
##     point-sharing, no negation or shift
##     if a = [2, 3, 4, 5, 6], then ts_buf will be:
##     [6, 5, 4, 3, (2, 3, 4, 5, 6,) 5, 4, 3, 2, ((2))]
    ts_buf[...,To-1:2*To-1] = a
    # imaginary part of ts is odd.. ts[-t] = -ts[t]
    ts_buf[...,:To-1] = np.conj(reverse(a[...,1:]))
    #ts_buf[...,:To-1] = reverse(a[...,1:])
    ts_buf[...,2*To-1:3*To-2] = reverse(a[...,:To-1])
    if To%2: ts_buf[...,-1] = ts_buf[...,-2]
    return ts_buf    

def subsampInterp(ts, c, axis=-1):
    """Makes a subsample sinc interpolation on an N-D array in a given axis.

    This uses sinc interpolation to return ts(t-a), where a is related to
    the factor c as described below.

    ts : the N-D array with a time series in "axis"
    c : the fraction of the sampling interval such that a = c*dt -- note that
      c is only useful in the interval (0, 1]
    axis : specifies the axis of the time dimension
    """
    To = ts.shape[axis]

    # find ramps and subtract them
    ramps = np.empty_like(ts)
    rax_shape = [1]*len(ts.shape)
    rax_shape[axis] = To
    rax = np.arange(To)
    rax.shape = rax_shape
    (mre,b,r) = lin_regression(ts.real, axis=axis)
    ramps.real[:] = rax*mre
    if ts.dtype.type in np.sctypes['complex']:
        (mim,b,r) = lin_regression(ts.imag, axis=axis)
        ramps.imag[:] = (rax*mim)
    np.subtract(ts, ramps, ts)
    # find biases and subtract them
    ts_mean = ts.mean(axis=axis)
    if len(ts_mean.shape):
        mean_shape = list(ts.shape)
        mean_shape[axis] = 1
        ts_mean.shape = tuple(mean_shape)
    np.subtract(ts, ts_mean, ts)
    
    # put time series in the last dimension
    if axis != -1:
        ts = np.swapaxes(ts, axis, -1)
    ts_buf = circularize(ts)
##     ts_buf = ts.copy()
    T = ts_buf.shape[-1]
    Fn = T/2 + 1
    # make sure the interpolating filter's dtype is complex!
    filter_dtype = np.dtype(ts.dtype.char.upper())
    phs_shift = np.empty((T,), filter_dtype)
    phs_shift[:Fn] = np.exp(-2.j*np.pi*c*np.arange(Fn)/float(T))
    phs_shift[Fn:] = np.conjugate(reverse(phs_shift[1:T-(Fn-1)]))
    fft(ts_buf, shift=False, inplace=True)
    np.multiply(ts_buf, phs_shift, ts_buf)
    ifft(ts_buf, shift=False, inplace=True)

    ts[:] = ts_buf[...,To-1:2*To-1]
##     ts[:] = ts_buf[:]
    del ts_buf
    if axis != -1:
        ts = np.swapaxes(ts, axis, -1)

    # add back biases and analytically interpolated ramps
    np.subtract(rax, c, rax)
    #np.add(rax, c, rax)
    ramps.real[:] = rax*mre
    if ts.dtype.type in np.sctypes['complex']:
        ramps.imag[:] = rax*mim
    np.add(ts, ramps, ts)
    np.add(ts, ts_mean, ts)
    
def subsampInterp_regular(ts, c, axis=-1):
    """Makes a subsample sinc interpolation on an N-D array in a given axis.

    This uses sinc interpolation to return ts(t-a), where a is related to
    the factor c as described below.

    ts : the N-D array with a time series in "axis"
    c : the fraction of the sampling interval such that a = c*dt -- note that
      c is only useful in the interval (0, 1]
    axis : specifies the axis of the time dimension
    """
    To = ts.shape[axis]
    
    # put time series in the last dimension
    if axis != -1:
        ts = np.swapaxes(ts, axis, -1)
##     ts_buf = circularize(ts)
    ts_buf = ts.copy()    
    T = ts_buf.shape[-1]
    Fn = T/2 + 1
    # make sure the interpolating filter's dtype is complex!
    filter_dtype = np.dtype(ts.dtype.char.upper())
    phs_shift = np.empty((T,), filter_dtype)
    phs_shift[:Fn] = np.exp(-2.j*np.pi*c*np.arange(Fn)/float(T))
    phs_shift[Fn:] = np.conjugate(reverse(phs_shift[1:T-(Fn-1)]))
    fft(ts_buf, shift=False, inplace=True)
    np.multiply(ts_buf, phs_shift, ts_buf)
    ifft(ts_buf, shift=False, inplace=True)

##     ts[:] = ts_buf[...,To-1:2*To-1]
    ts[:] = ts_buf[:]
    del ts_buf
    if axis != -1:
        ts = np.swapaxes(ts, axis, -1)

def subsampInterp_TD(ts, c, axis=0):
    T = ts.shape[axis]

    # find ramps and subtract them
    ramps = np.empty_like(ts)
    rax_shape = [1]*len(ts.shape)
    rax_shape[axis] = T
    rax = np.arange(T)
    rax.shape = rax_shape
    (mre,b,r) = lin_regression(ts.real, axis=axis)
    ramps.real[:] = rax*mre
    if ts.dtype.type in np.sctypes['complex']:
        (mim,b,r) = lin_regression(ts.imag, axis=axis)
        ramps.imag[:] = (rax*mim)
    np.subtract(ts, ramps, ts)
    # find biases and subtract them
    ts_mean = ts.mean(axis=axis)
    if len(ts_mean.shape):
        mean_shape = list(ts.shape)
        mean_shape[axis] = 1
        ts_mean.shape = tuple(mean_shape)
    np.subtract(ts, ts_mean, ts)

    
    if axis != 0:
        ts = np.swapaxes(ts, axis, 0)
    snc_ax = np.arange(T)
    ts_shape = ts.shape
    ts.shape = (T, -1)
    snc_kern = np.sinc(snc_ax[None,:] - snc_ax[:,None] + c)
    ts_tmp = np.dot(snc_kern, ts)
    ts[:] = ts_tmp
    ts.shape = ts_shape
    del ts_tmp
    if axis != 0:
        ts = np.swapaxes(ts, axis, 0)
    
    # add back biases and analytically interpolated ramps
    np.subtract(rax, c, rax)
    #np.add(rax, c, rax)
    ramps.real[:] = rax*mre
    if ts.dtype.type in np.sctypes['complex']:
        ramps.imag[:] = rax*mim
    np.add(ts, ramps, ts)
    np.add(ts, ts_mean, ts)


    
def gen_mixed_freqs(over_sample=1):
    L = 400*over_sample # samp-rate = L Hz, so max freq = L/2 Hz
    maxfreq = 200
    Nfreq = 5
    z = np.zeros((L,), np.complex64)
    freqs = np.random.randint(-maxfreq, maxfreq, Nfreq)
    freqs = np.random.normal(size=Nfreq, scale=maxfreq/4.)
    np.clip(freqs, -maxfreq, maxfreq, out=freqs)
    coefs = np.random.normal(size=Nfreq) + 1j*np.random.normal(size=Nfreq)
    for f,c in zip(freqs,coefs):
        np.add(z, c.real*np.cos(2*np.pi*f*np.arange(L)/L) +
               1j*c.imag*np.sin(2*np.pi*f*np.arange(L)/L),
               z)
    # add a DC with stronger coef
    c0 = np.random.normal(size=1, scale=40)
    np.add(z, c0, z)
    # add subtle ramps
    r = np.random.normal(size=2, scale=.015)
    np.add(z, np.arange(L) * (r[0] + 1j*r[1]), z)
    return z
    
class FixTimeSkew (Operation):
    """
    Use sinc interpolation to shift slice data back-in-time to a point
    corresponding to the beginning of acquisition.
    """

    params=(
        Parameter(name="data_space", type="str", default="kspace",
            description="""
    name of space to run op: kspace or imspace."""),
        )
    
    @ChannelIndependentOperation
    def run(self, image):
##         if not verify_scanner_image(self, image):
##             return
        if image.tdim < 2:
            self.log("Cannot interpolate with only one volume")
            return

        nslice = image.kdim
        # slice acquisition can be in some nonlinear order
        # eg: Varian data is acquired in an order like this:
        # [19,17,15,13,11, 9, 7, 5, 3, 1,18,16,14,12,10, 8, 6, 4, 2, 0,]
        # So the phase shift factor c for spatially ordered slices should go:
        # [19/20., 9/20., 18/20., 8/20., ...]

        # --- slices in order of acquistion ---
        acq_order = image.acq_order
        # --- shift factors indexed slice number ---
        shifts = np.array([np.nonzero(acq_order==s)[0] for s in range(nslice)])

        if self.data_space == "imspace":
            image.setData(np.abs(image[:]).astype(np.float32))

        # image-space magnitude interpolation can't be
        # multisegment sensitive, so do it as if it's 1-seg
        if image.nseg == 1 or self.data_space=="imspace":
            for s in range(nslice):
                c = float(shifts[s])/float(nslice)
                #sl = (slice(0,nvol), slice(s,s+1), slice(0,npe), slice(0,nfe))
                subsampInterp(image[:,s,:,:], c, axis=0)
        else:
            # get the appropriate slicing for sampling type
            #sl1 = self.segn(image,0)
            #sl2 = self.segn(image,1)
            sl1 = image.seg_slicing(0)
            sl2 = image.seg_slicing(1)
            for s in range(nslice):
                # want to shift seg1 forward temporally and seg2 backwards--
                # have them meet halfway (??)
                c1 = -(nslice-shifts[s]-0.5)/float(nslice)
                c2 = (shifts[s]+0.5)/float(nslice)
                # interpolate for each segment        
                subsampInterp_regular(image[:,s,sl1,:], c1, axis=0)
                subsampInterp_regular(image[:,s,sl2,:], c2, axis=0)
        
    
