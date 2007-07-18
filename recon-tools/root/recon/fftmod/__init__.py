import numpy as N

try:
    from _fftmod import fft1d, fft2d
except ImportError:
    raise ImportError("Please compile the fftmod extension to use this module")


def fft1_generic(a, direction):
    return fft1d(a, direction)

def fft2_generic(a, direction):
    return fft2d(a, direction)

def fft1(a, axis=-1):
    if axis != -1:
        a = N.swapaxes(a, axis, -1)
    b = fft1_generic(a, -1)
    if axis != -1:
        a = N.swapaxes(a, axis, -1)
        b = N.swapaxes(b, axis, -1)
    return b

def ifft1(a, axis=-1):
    if axis != -1:
        a = N.swapaxes(a, axis, -1)
    b = fft1_generic(a, +1)
    if axis != -1:
        a = N.swapaxes(a, axis, -1)
        b = N.swapaxes(b, axis, -1)
    return b

#must be done on axes=(-2,-1)
def fft2(a):
    return fft2_generic(a, -1)

def ifft2(a):
    return fft2_generic(a, +1)
    
        
