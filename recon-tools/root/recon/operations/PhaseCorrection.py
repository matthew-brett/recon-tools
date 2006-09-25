from Numeric import empty
from pylab import mlab, angle, cos, sin, Float, Complex32, asarray
from recon.operations import Operation
from recon.util import shift, ifft, apply_phase_correction

##############################################################################
class PhaseCorrection (Operation):

    #-------------------------------------------------------------------------
    def run(self, image):
        if not image.ref_data:
            self.log("No reference data, nothing to do.")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        # phase angle of inverse fft'd reference volume
        ref_phs = angle(ifft(image.ref_data[0]))
        
        # apply correction to image data
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image.data = apply_phase_correction(image.data, -ref_phs)
        else:
            for dvol in image.data:
                dvol[:] = apply_phase_correction(dvol, -ref_phs)