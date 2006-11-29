from recon.util import fft2d
from recon.operations import Operation

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
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = fft2d(image[:]).astype(image[:].typecode())
        else:
            for vol in image.data:
                vol[:] = fft2d(vol).astype(vol.typecode())

