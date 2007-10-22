from numpy.testing import *
set_package_path()
from fftmod import fft1, ifft1, fft2, ifft2
from recon import util
import numpy as N

def direct_FFT(v):
    assert v.dtype.char.isupper()
    L = v.shape[-1]
    basis = N.exp(N.outer(N.arange(L), -2.j*N.pi*N.arange(L)/L)).astype(v.dtype)
    return N.dot(basis, v)

def direct_FFT_centered(v):
    assert v.dtype.char.isupper()
    L = v.shape[-1]
    grid = N.linspace(-L/2,L/2,num=L,endpoint=False)
    basis = N.exp(N.outer(grid, -2.j*N.pi*grid/L)).astype(v.dtype)
    return N.dot(basis, v)

class test_FFTs(NumpyTestCase):

    def check_fft1(self, level=1):
        # a 32Hz complex exponential (sampling rate = 128Hz)
        sgrid = N.linspace(0, 1, num=128, endpoint=False)
        c = N.exp(2.j*N.pi*32*sgrid).astype(N.complex64)
        z = N.exp(2.j*N.pi*32*sgrid).astype(N.complex128)
        C1 = fft1(c, shift=False)
        C2 = fft1(c, shift=True)
        Z1 = fft1(z, shift=False)
        Z2 = fft1(z, shift=True)
        # analytically, the DFT indexed from 0,127 of s
        # is a weighted delta at k=32
        C1_a = N.zeros(128, N.complex64)
        Z1_a = N.zeros(128, N.complex128)
        C1_a[32] = 128.
        Z1_a[32] = 128.
        assert_array_almost_equal(C1, C1_a, decimal=6)
        assert_array_almost_equal(Z1, Z1_a, decimal=12)
        # compare with FFTs given with spectrum centered around L/2
        util.shift(C1_a, 64)
        util.shift(Z1_a, 64)
        assert_array_almost_equal(C2, C1_a, decimal=6)
        assert_array_almost_equal(Z2, Z1_a, decimal=12)

    
    def test_fft1(self, level=1):
        for dt, dec in ((N.complex64, 5), (N.complex128, 10)):
            sig = (N.random.rand(128) + 1.j*N.random.rand(128)).astype(dt)
            SIG = direct_FFT(sig)
            SIGc = direct_FFT_centered(sig)
            assert_array_almost_equal(fft1(sig,shift=False), SIG, decimal=dec)
            assert_array_almost_equal(fft1(sig,shift=True), SIGc, decimal=dec)
        
        
    
