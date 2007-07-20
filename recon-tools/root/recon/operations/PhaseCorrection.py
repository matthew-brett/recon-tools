import numpy as N
from recon.operations import Operation
from recon.util import shift, ifft, apply_phase_correction

##############################################################################
class PhaseCorrection (Operation):
    """
    Perform a FE-direction phase correction based on a reference volume scan
    to reduce ghosting errors.
    """

    #-------------------------------------------------------------------------
    def run(self, image):
        if not hasattr(image, 'ref_data'):
            self.log("No reference data, nothing to do.")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        # phase angle of inverse fft'd reference volume
        ref_phs = N.angle(ifft(image.ref_data[0]))
        
        # apply correction to image data
        from recon.tools import Recon
        phase = N.exp(-1.j*ref_phs)
        if Recon._FAST_ARRAY:
            image[:] = apply_phase_correction(image[:], phase)
        else:
            for dvol in image:
                dvol[:] = apply_phase_correction(dvol[:], phase)
