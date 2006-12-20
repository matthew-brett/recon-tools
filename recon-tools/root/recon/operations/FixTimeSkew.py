from FFT import fft, inverse_fft
import Numeric as N

from recon.operations import Operation, Parameter, verify_scanner_image
from recon.util import reverse

def circularize(a):
    To = a.shape[-1]
    T = 3*To -2 + To%2
    #T = 3*To + To%2
    ts_buf = N.empty(a.shape[:-1]+ (T,), N.Complex)
#### point-sharing, no negation or shift
#### if a = [2, 3, 4, 5, 6], then ts_buf will be:
#### [6, 5, 4, 3, (2, 3, 4, 5, 6,) 5, 4, 3, 2, ((2))]
    ts_buf[...,To-1:2*To-1] = a
    ts_buf[...,:To-1] = reverse(a[...,1:])
    ts_buf[...,2*To-1:3*To-2] = reverse(a[...,:To-1])

############ OTHER METHODS ##################################    
#### point-sharing, triple buffered, reversed and negated
##     ts_buf[...,To-1:2*To-1] = a
##     ts_buf[...,:To-1] = -reverse(a[...,1:]) + \
##                       N.array(3*a[...,1] - a[...,2])[...,N.NewAxis]
##     ts_buf[...,2*To-1:3*To-2] = -reverse(a[...,:To-1]) + \
##                         N.array(3*a[...,-2] - a[...,-3])[...,N.NewAxis]
#### double buffered, reversed, negated, no point sharing
##     ts_buf[...,:To] = a
##     ts_buf[...,To:] = -lutil.reverse(a) + N.array(3*a[...,0]-a[...,1])[...,N.NewAxis]
## original (triple buffered, no point sharing, negated and reversed
##     ts_buf[...,To:2*To] = a
##     ts_buf[...,:To] = -reverse(a) + \
##                       N.array(3*a[...,0] - a[...,1])[...,N.NewAxis]
##     ts_buf[...,2*To:3*To] = -reverse(a) + \
##                         N.array(3*a[...,-1] - a[...,-2])[...,N.NewAxis]
#############################################################
    
    if To%2: ts_buf[...,-1] = ts_buf[...,-2]
    return ts_buf

    
def subsampInterp(ts, c, axis=-1):
    # convert arbitrarily shaped ts into a 2d array
    # with time series in last dimension
    To = ts.shape[axis]
    if axis != -1:
        ts = N.swapaxes(ts, axis, -1)

    ts_buf = circularize(ts)
    T = ts_buf.shape[-1]
    Fn = int(T/2.) + 1
    phs_shift = N.empty((T,), N.Complex)
    phs_shift[:Fn] = N.exp(-2.j*N.pi*c*N.arange(Fn)/float(T))
    phs_shift[Fn:] = N.conjugate(reverse(phs_shift[1:Fn-1]))

    ts[:] = inverse_fft(fft(ts_buf)*phs_shift)[...,To-1:2*To-1].astype(ts.typecode())

    if axis != -1: ts = N.swapaxes(ts, axis, -1)

class FixTimeSkew (Operation):
    """
    Use sinc interpolation to shift slice data back-in-time to a point
    corresponding to the beginning of acquisition.
    """

    params=(
        Parameter(name="data_space", type="str", default="kspace",
                  description="name of space to run op: kspace or imspace"),
        )

    def run(self, image):
        if not verify_scanner_image(self, image): return
        nslice = image.zdim
        # slice acquisition can be in some nonlinear order
        # eg: Varian data is acquired in an order like this:
        # [19,17,15,13,11, 9, 7, 5, 3, 1,18,16,14,12,10, 8, 6, 4, 2, 0,]
        # So the phase shift factor c for spatially ordered slices should go:
        # [19/20., 9/20., 18/20., 8/20., ...]

        # --- slices in order of acquistion ---
        acq_order = image.acq_order
        # --- shift factors indexed slice number ---
        shifts = N.array([N.nonzero(acq_order==s)[0] for s in range(nslice)])

        # I don't like this kludge..
        if self.data_space == "imspace":
            image.data = abs(image[:]).astype('f')

        # image-space magnitude interpolation can't be
        # multisegment sensitive, so do it as if it's 1-seg
        if image.nseg == 1 or self.data_space=="imspace":
            for s in range(nslice):
                c = float(shifts[s])/float(nslice)
                #sl = (slice(0,nvol), slice(s,s+1), slice(0,npe), slice(0,nfe))
                subsampInterp(image[:,s,:,:], c, axis=0)
        else:
            for s in range(nslice):
                # want to shift seg1 forward temporally and seg2 backwards--
                # have them meet halfway (??)
                c1 = -(nslice-shifts[s]-0.5)/float(nslice)
                c2 = (shifts[s]+0.5)/float(nslice)
                # get the appropriate slicing for sampling type
                sl1 = self.segn(image,0)
                sl2 = self.segn(image,1)
                # interpolate for each segment        
                subsampInterp(image[:,s,sl1,:], c1, axis=0)
                subsampInterp(image[:,s,sl2,:], c2, axis=0)

                
    def segn(self, image, n):
        # this is very very limited to 2-shot trajectories!
        npe = image.shape[-2]
        if image.sampstyle == "centric":
            return slice(n*npe/2,npe/(2-n))
        else: return slice(n,npe,2)
    
