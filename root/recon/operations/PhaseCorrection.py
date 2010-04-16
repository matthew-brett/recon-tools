import numpy as np
from recon.operations import Operation, Parameter, ChannelIndependentOperation
from recon.util import shift, ifft, apply_phase_correction

##############################################################################
class PhaseCorrection (Operation):
    """
    Perform a FE-direction phase correction based on a reference volume scan
    to reduce ghosting errors.
    """

    #-------------------------------------------------------------------------
    @ChannelIndependentOperation
    def run(self, image):
        if not hasattr(image, 'ref_data'):
            self.log("No reference data, nothing to do.")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        # phase angle of inverse fft'd reference volume
        iref_data = ifft(image.ref_data[0])
        ref_phs = np.angle(iref_data)

        # apply correction to image data
        from recon.tools import Recon
        phase = np.exp(-1.j*ref_phs).astype(image[:].dtype)
        if Recon._FAST_ARRAY:
            apply_phase_correction(image[:], phase)
        else:
            for dvol in image:
                apply_phase_correction(dvol[:], phase)
