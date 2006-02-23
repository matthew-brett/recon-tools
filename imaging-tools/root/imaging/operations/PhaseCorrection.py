from Numeric import empty
from FFT import inverse_fft
from pylab import mlab, angle, fft, cos, sin, Float, Complex32, asarray
from imaging.operations import Operation
from imaging.util import shift, apply_phase_correction


##############################################################################
class PhaseCorrection (Operation):

    #-------------------------------------------------------------------------
    def run(self, image):
        if not image.ref_data:
            self.log("No reference data, nothing to do.")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        refVol = image.ref_data[0]                             
        # phase angle of inverse fft'd reference volume
        ref_phs = angle(inverse_fft(refVol))
        
        # apply correction to image data
        image.data = apply_phase_correction(image.data, -ref_phs)

        # put back in increasing frequency domain
        shift(image.data, 0, image.data.shape[-1]/2)
