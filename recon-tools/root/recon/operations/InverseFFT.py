from recon.util import ifft2d
from recon.operations import Operation

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
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = ifft2d(image[:]).astype(image[:].typecode())
        else:
            for vol in image.data:
                vol[:] = ifft2d(vol).astype(vol.typecode())
