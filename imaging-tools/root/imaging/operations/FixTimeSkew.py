from FFT import fft, inverse_fft
from pylab import arange, exp, pi, take, conjugate, empty, Complex, Complex32
from imaging.operations import Operation
from imaging.util import reverse

def subsampInterp(ts, c):
    T = 3*len(ts)
    Fn = int(T/2.) + 1
    ts_buf = empty((T,), Complex)
    ts_buf[len(ts):2*len(ts)] = ts
    ts_buf[:len(ts)] = -reverse(ts) + 2*ts[0] - (ts[1]-ts[0])
    ts_buf[2*len(ts):] = -reverse(ts) + 2*ts[-1] - (ts[-2]-ts[-1])
    phs_shift = empty((T,), Complex)
    phs_shift[:Fn] = exp(-2.j*pi*c*arange(Fn)/float(T))
    phs_shift[Fn:] = conjugate(reverse(phs_shift[1:Fn-1]))
    return inverse_fft(fft(ts_buf)*phs_shift)[len(ts):2*len(ts)]

class FixTimeSkew (Operation):

    def run(self, image):
        (nvol, nslice, npe, nfe) = image.data.shape
        if image.nseg == 1:
            for s in range(nslice):
                c = float(s)/float(nslice)
                for p in range(npe):
                    for r in range(nfe):
                        ts = image.data[:,s,p,r]
                        image.data[:,s,p,r] = subsampInterp(ts,c).astype(Complex32)
        else:
            for s in range(nslice):
                # want to shift seg1 forward temporally and seg2 backwards--
                # have them meet halfway (??)
                c1 = -(S-s-0.5)/float(nslice)
                c2 = (s+0.5)/float(nslice)
                for r in range(nfe):
                    for p in self.segn(image,0):
                        ts = image.data[:,s,p,r]
                        image.data[:,s,p,r] = subsampInterp(ts,c1).astype(Complex32)
                    for p in self.segn(image,1):
                        ts = image.data[:,s,p,r]
                        image.data[:,s,p,r] = subsampInterp(ts,c2).astype(Complex32)

    def segn(self, image, n):
        npe = image.data.shape[-2]
        if image.petable_name.find('cen') > 0 or \
           image.petable_name.find('alt') > 0:
            return arange(npe/2)+n*npe/2
        else: return arange(n,npe,2)
    
