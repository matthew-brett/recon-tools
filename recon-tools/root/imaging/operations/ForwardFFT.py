from FFT import fft2d
from pylab import Complex32
from imaging.util import checkerboard
from imaging.operations import Operation

##############################################################################
class ForwardFFT (Operation):
    """
    Perform an inverse 2D fft on each slice of each k-space volume.
    For the sake of completeness, this inverts the (S_k)(FT)(S_i)
    sequence of operations in InverseFFT by (S_i)^-1(FT)^-1(S_k)^-1.
    A Shift in one space is applied in the complementary space by
    way of amplitude-modulation/phase-adjustment
    """

    #-------------------------------------------------------------------------
    def run(self, image):
        ydim, xdim = image.data.shape[-2:]
        mask = checkerboard(ydim, xdim)
        from imaging.tools import Recon
        if Recon._FAST_ARRAY:
            image.data[:] = (mask * fft2d(mask * image.data)).astype(Complex32)
        else:
            for vol in image.data:
                vol[:] = (mask * fft2d(mask * vol)).astype(Complex32)

