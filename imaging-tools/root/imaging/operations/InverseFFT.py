from FFT import inverse_fft2d
from pylab import Complex32
from imaging.util import shift
from imaging.operations import Operation

##############################################################################
class InverseFFT (Operation):
    "Perform an inverse 2D fft on each slice of each k-space volume."

    #-------------------------------------------------------------------------
    def run(self, image):
        n_pe, n_fe = image.data.shape[-2:]
        for volume in image.data:
            for slice in volume:
                slice[:] = (image._2D_checkerboard * \
                            inverse_fft2d(image._2D_checkerboard*slice)).astype(Complex32)
