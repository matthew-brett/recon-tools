from FFT import fft, inverse_fft
from pylab import arange, exp, pi, take, conjugate, empty, Complex, Complex32,\
     swapaxes, product, NewAxis, reshape, squeeze
from recon.operations import Operation
from recon.util import reverse

def subsampInterp(ts, c, axis=-1):
    # convert arbitrarily shaped ts into a 2d array
    # with time series in last dimension
    To = ts.shape[axis]
    dimlist = list(ts.shape)
    dimlist.remove(To)
    if axis != -1: ts = swapaxes(ts, axis, -1)
    T = 3*To
    Fn = int(T/2.) + 1
    ts_buf = empty((dimlist[-1],)+tuple(dimlist[:-1])+ (T,), Complex)
    ts_buf[...,To:2*To] = ts
    ts_buf[...,:To] = -reverse(ts) + \
                      (2*ts[...,0] - (ts[...,1]-ts[...,0]))[...,NewAxis]
    ts_buf[...,2*To:] = -reverse(ts) + \
                        (2*ts[...,-1] - (ts[...,-2]-ts[...,-1]))[...,NewAxis]
    phs_shift = empty((T,), Complex)
    phs_shift[:Fn] = exp(-2.j*pi*c*arange(Fn)/float(T))
    phs_shift[Fn:] = conjugate(reverse(phs_shift[1:Fn-1]))
    ts[:] = inverse_fft(fft(ts_buf)*phs_shift)[...,To:2*To].astype(ts.typecode())
    if axis != -1: ts = swapaxes(ts, axis, -1)

class FixTimeSkew (Operation):

    def run(self, image):
        (nvol, nslice, npe, nfe) = image.data.shape
        if image.nseg == 1:
            for s in range(nslice):
                c = float(s)/float(nslice)
                #sl = (slice(0,nvol), slice(s,s+1), slice(0,npe), slice(0,nfe))
                subsampInterp(image.data[:,s,:,:], c, axis=0)
        else:
            for s in range(nslice):
                # want to shift seg1 forward temporally and seg2 backwards--
                # have them meet halfway (??)
                c1 = -(nslice-s-0.5)/float(nslice)
                c2 = (s+0.5)/float(nslice)
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
    
