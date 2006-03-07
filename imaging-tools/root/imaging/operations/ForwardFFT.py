from FFT import fft2d
from pylab import Complex32
from imaging.util import shift
from imaging.operations import Operation

##############################################################################
class ForwardFFT (Operation):
    """
    Perform an inverse 2D fft on each slice of each k-space volume.
    For the sake of completeness, this inverses the (S_k)(FT)(S_i)
    sequence of operations in InverseFFT by (S_i)^-1(FT)^-1(S_k)^-1
    """

    #-------------------------------------------------------------------------
    def run(self, image):
        ydim, xdim = image.data.shape[-2:]
        for volume in image.data:
            for slice in volume:
                slice[:] = (image._2D_checkerboard * \
                            fft2d(image._2D_checkerboard*slice)).astype(Complex32)
