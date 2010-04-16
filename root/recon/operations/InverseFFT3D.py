from numpy.fft import ifftn
from recon.util import checkercube
from recon.operations import Operation, ChannelIndependentOperation

##############################################################################
class InverseFFT3D (Operation):
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
        nslice, n_pe, n_fe = image.shape[-3:]
        mask = checkercube(nslice, n_pe, n_fe)
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = mask*ifftn(mask*image[:], axes=[-3,-2,-1])
        else:
            for vol in image:
                vol[:] = mask*ifftn(mask*vol[:])
