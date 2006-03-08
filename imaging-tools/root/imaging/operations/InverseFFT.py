from FFT import inverse_fft2d
from pylab import Complex32
from imaging.util import checkerboard
from imaging.operations import Operation

##############################################################################
class InverseFFT (Operation):
    """
    Perform an inverse 2D fft on each slice of each k-space volume.
    This operation additionally must translate our interpretation
    of the data with indices ranging in [-pi,pi) to the form used
    by the FFT algorithm (from [0, pi] U (-pi,0)). Then at the
    output, it must translate a "time scale" of [0,T) back to our
    scale of [-T/2,T/2). These shifts are performed in the inverse
    spaces as phase-adjustment/amplitude-demodulation
    """

    #-------------------------------------------------------------------------
    def run(self, image):
        n_pe, n_fe = image.data.shape[-2:]
        mask = checkerboard(n_pe, n_fe)
        for volume in image.data:
            for slice in volume:
                slice[:] = (mask * inverse_fft2d(mask * slice)).astype(Complex32)
