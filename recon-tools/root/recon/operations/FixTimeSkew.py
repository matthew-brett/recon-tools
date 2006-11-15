from FFT import fft, inverse_fft
from pylab import arange, exp, pi, take, conjugate, empty, Complex, Complex32,\
     swapaxes, product, NewAxis, reshape, squeeze, array, find
from recon.operations import Operation, Parameter
from recon.util import reverse

def circularize(a):
    To = a.shape[-1]
    T = 3*To -2 + To%2
    ts_buf = empty(a.shape[:-1]+ (T,), Complex)
#### point-sharing
    ts_buf[...,To-1:2*To-1] = a
    ts_buf[...,:To-1] = -reverse(a[...,1:]) + \
                      array(3*a[...,1] - a[...,2])[...,NewAxis]
    ts_buf[...,2*To-1:3*To-2] = -reverse(a[...,:To-1]) + \
                        array(3*a[...,-2] - a[...,-3])[...,NewAxis]
#### double buffered and reversed
##     ts_buf[...,:To] = a
##     ts_buf[...,To:] = -lutil.reverse(a) + array(3*a[...,0]-a[...,1])[...,NewAxis]
#### original
##     ts_buf[...,To:2*To] = a
##     ts_buf[...,:To] = -lutil.reverse(a) + \
##                       array(3*a[...,0] - a[...,1])[...,NewAxis]
##     ts_buf[...,2*To:3*To] = -lutil.reverse(a) + \
##                         array(3*a[...,-1] - a[...,-2])[...,NewAxis]
#### point-sharing, no negation or shift
##     ts_buf[...,To-1:2*To-1] = a
##     ts_buf[...,:To-1] = lutil.reverse(a[...,:To-1])
##     ts_buf[...,2*To-1:3*To-1] = lutil.reverse(a[...,:To-1])
    if To%2: ts_buf[...,-1] = ts_buf[...,-2]
    return T, ts_buf

    
def subsampInterp(ts, c, axis=-1):
    # convert arbitrarily shaped ts into a 2d array
    # with time series in last dimension
    To = ts.shape[axis]
    if axis != -1:
        ts = swapaxes(ts, axis, -1)

    T, ts_buf = circularize(ts)

    Fn = int(T/2.) + 1
    phs_shift = empty((T,), Complex)
    phs_shift[:Fn] = exp(-2.j*pi*c*arange(Fn)/float(T))
    phs_shift[Fn:] = conjugate(reverse(phs_shift[1:Fn-1]))

    ts[:] = inverse_fft(fft(ts_buf)*phs_shift)[...,To-1:2*To-1].astype(ts.typecode())

    if axis != -1: ts = swapaxes(ts, axis, -1)

class FixTimeSkew (Operation):
    """
    Use sinc interpolation to shift slice data back-in-time to a point
    corresponding to the beginning of acquisition.
    """

    params=(
        Parameter(name="flip_slices", type="bool", default=False,
                  description="Flip slices during reordering."),
        )

    def run(self, image):
        (nvol, nslice, npe, nfe) = image.data.shape
        # slice acquisition can be in some nonlinear order
        # eg: Varian data is acquired in an order like this:
        # [19,17,15,13,11, 9, 7, 5, 3, 1,18,16,14,12,10, 8, 6, 4, 2, 0,]
        # So the phase shift factor c for spatially ordered slices should go:
        # [19/20., 9/20., 18/20., 8/20., ...]

        # --- slices in order of acquistion ---
        acq_order = image.acq_order
        # --- shift factors indexed slice number ---
        shifts = array([find(acq_order==s)[0] for s in range(nslice)])
        if self.flip_slices: shifts = reverse(shifts)
        
        if image.nseg == 1:
            for s in range(nslice):
                c = float(shifts[s])/float(nslice)
                #sl = (slice(0,nvol), slice(s,s+1), slice(0,npe), slice(0,nfe))
                subsampInterp(image.data[:,s,:,:], c, axis=0)
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
                subsampInterp(image.data[:,s,sl1,:], c1, axis=0)
                subsampInterp(image.data[:,s,sl2,:], c2, axis=0)

    def segn(self, image, n):
        # this is very very limited to 2-shot trajectories!
        npe = image.data.shape[-2]
        if image.petable_name.find('cen') > 0 or \
           image.petable_name.find('alt') > 0:
            return slice(n*npe/2,npe/(2-n))
        else: return slice(n,npe,2)
    
