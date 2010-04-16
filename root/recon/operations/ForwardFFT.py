from recon.fftmod import fft2
from recon.operations import Operation, ChannelIndependentOperation

##############################################################################
class ForwardFFT (Operation):
    """
    Perform a forward 2D fft on each slice of each k-space volume.
    For the sake of completeness, this inverts the index-reordering
    sequence of operations in InverseFFT. A Shift in one space is applied
    in the complementary space by way of amplitude-modulation.
    """

    #-------------------------------------------------------------------------
    @ChannelIndependentOperation
    def run(self, image):
        fft2(image[:], inplace=True)

