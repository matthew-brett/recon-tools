import numpy as N
from recon.fftmod import fft1 as fft, ifft1 as ifft
from recon.operations import Operation, Parameter, verify_scanner_image
from recon.util import reverse

def circularize(a):
    To = a.shape[-1]
    T = 3*To -2 + To%2
    buf_dtype = N.dtype(a.dtype.char.upper())
    ts_buf = N.empty(a.shape[:-1]+ (T,), buf_dtype)
##     point-sharing, no negation or shift
##     if a = [2, 3, 4, 5, 6], then ts_buf will be:
##     [6, 5, 4, 3, (2, 3, 4, 5, 6,) 5, 4, 3, 2, ((2))]
    ts_buf[...,To-1:2*To-1] = a
    ts_buf[...,:To-1] = reverse(a[...,1:])
    ts_buf[...,2*To-1:3*To-2] = reverse(a[...,:To-1])
    if To%2: ts_buf[...,-1] = ts_buf[...,-2]
    return ts_buf    

def subsampInterp(ts, c, axis=-1):
    # put time series in the last dimension
    To = ts.shape[axis]
    if axis != -1:
        ts = N.swapaxes(ts, axis, -1)
    ts_buf = circularize(ts)
    T = ts_buf.shape[-1]
    Fn = int(T/2.) + 1
    # make sure the interpolating filter's dtype is complex!
    filter_dtype = N.dtype(ts.dtype.char.upper())
    phs_shift = N.empty((T,), filter_dtype)
    phs_shift[:Fn] = N.exp(-2.j*N.pi*c*N.arange(Fn)/float(T))
    phs_shift[Fn:] = N.conjugate(reverse(phs_shift[1:Fn-1]))
    ts[:] = ifft(fft(ts_buf,shift=False)*phs_shift,shift=False)[...,To-1:2*To-1]
    if axis != -1: ts = N.swapaxes(ts, axis, -1)

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

    def run(self, image):
        if not verify_scanner_image(self, image):
            return
        if image.ndim < 4:
            self.log("Cannot interpolation with only one volume")
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
        shifts = N.array([N.nonzero(acq_order==s)[0] for s in range(nslice)])

        if self.data_space == "imspace":
            image.setData(N.abs(image[:]).astype(N.float32))

        # image-space magnitude interpolation can't be
        # multisegment sensitive, so do it as if it's 1-seg
        if image.nseg == 1 or self.data_space=="imspace":
            for s in range(nslice):
                c = float(shifts[s])/float(nslice)
                #sl = (slice(0,nvol), slice(s,s+1), slice(0,npe), slice(0,nfe))
                subsampInterp(image[:,s,:,:], c, axis=0)
        else:
            # get the appropriate slicing for sampling type
            sl1 = self.segn(image,0)
            sl2 = self.segn(image,1)
            for s in range(nslice):
                # want to shift seg1 forward temporally and seg2 backwards--
                # have them meet halfway (??)
                c1 = -(nslice-shifts[s]-0.5)/float(nslice)
                c2 = (shifts[s]+0.5)/float(nslice)
                # interpolate for each segment        
                subsampInterp(image[:,s,sl1,:], c1, axis=0)
                subsampInterp(image[:,s,sl2,:], c2, axis=0)
        
    def segn(self, image, n):
        # this is very very limited to 2-shot trajectories!
        npe = image.shape[-2]
        if image.sampstyle == "centric":
            return slice(n*npe/2,npe/(2-n))
        else: return slice(n,npe,2)
    
