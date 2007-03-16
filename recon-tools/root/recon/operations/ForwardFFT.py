from recon.util import fft2d
from recon.operations import Operation

##############################################################################
class ForwardFFT (Operation):
    """
    Perform a forward 2D fft on each slice of each k-space volume.
    For the sake of completeness, this inverts the index-reordering
    sequence of operations in InverseFFT. A Shift in one space is applied
    in the complementary space by way of amplitude-modulation.
    """

    #-------------------------------------------------------------------------
    def run(self, image):
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = fft2d(image[:])
        else:
            for vol in image:
                vol[:] = fft2d(vol[:])

