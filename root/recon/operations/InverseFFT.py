from recon.fftmod import ifft2
from recon.operations import Operation, ChannelIndependentOperation
##############################################################################
class InverseFFT (Operation):
    """
    Perform an inverse 2D fft on each slice of each k-space volume.
    This operation additionally must translate our interpretation
    of the data with indices ranging in [-M,M-1] to the form used
    by the FFT algorithm (from [0, M-1] U [-M,-1]). Then at the
    output, it must translate an im-space index of [0,2Q-1] back to
    our scale of [-Q/2,Q/2-1]. These shifts are performed in the
    inverse spaces as amplitude-demodulation.
    """

    #-------------------------------------------------------------------------
    @ChannelIndependentOperation
    def run(self, image):
        ifft2(image[:], inplace=True)
