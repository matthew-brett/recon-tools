from Numeric import empty
from pylab import mlab, angle, cos, sin, Float, Complex32, asarray
from imaging.operations import Operation
from imaging.util import shift, ifft, apply_phase_correction, y_grating


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
        ref_phs = angle(ifft(image.ref_data[0], shift=True))
        
        # apply correction to image data
        image.data = apply_phase_correction(image.data, -ref_phs, shift=True)
