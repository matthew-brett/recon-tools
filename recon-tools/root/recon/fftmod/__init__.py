import numpy as N

try:
    from _fftmod import fft1d, fft2d
except ImportError:
    raise ImportError("Please compile the fftmod extension to use this module")


def fft1(a, axis=-1, shift=True):
    if axis != -1:
        a = N.swapaxes(a, axis, -1)
    b = fft1d(a, -1, shift)
    if axis != -1:
        a = N.swapaxes(a, axis, -1)
        b = N.swapaxes(b, axis, -1)
    return b

def ifft1(a, axis=-1, shift=True):
    if axis != -1:
        a = N.swapaxes(a, axis, -1)
    b = fft1d(a, +1, shift)
    if axis != -1:
        a = N.swapaxes(a, axis, -1)
        b = N.swapaxes(b, axis, -1)
    return b

#must be done on axes=(-2,-1)
def fft2(a, shift=True):
    return fft2d(a, -1, shift)

def ifft2(a, shift=True):
    return fft2d(a, +1, shift)
    
        
