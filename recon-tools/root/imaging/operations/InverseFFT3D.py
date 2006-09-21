from FFT import inverse_fftnd
from pylab import Complex32
from recon.util import checkercube
from recon.operations import Operation

##############################################################################
class InverseFFT3D (Operation):
    """
    Perform an inverse 3D fft on each volume of k-space data.
    This operation additionally must translate our interpretation
    of the data with indices ranging in [-pi,pi) to the form used
    by the FFT algorithm (from [0, pi] U (-pi,0)). Then at the
    output, it must translate a "time scale" of [0,T) back to our
    scale of [-T/2,T/2). These shifts are performed in the inverse
    spaces as phase-adjustment/amplitude-demodulation
    """

    #-------------------------------------------------------------------------
    def run(self, image):
        nslice, n_pe, n_fe = image.data.shape[-3:]
        mask = checkercube(nslice, n_pe, n_fe)
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image.data[:] = (mask*inverse_fftnd(mask*image.data,
                                                axes=[1,2,3])).astype(Complex32)
        else:
            for volume in image.data:
                volume[:] = (mask*inverse_fftnd(mask*volume)).astype(Complex32)
