"Applies a Balanced Phase Correction on data sets with two reference volumes"

from FFT import inverse_fft
from pylab import angle, conjugate, sin, cos, Complex32, fft
from imaging.operations import Operation
from imaging.util import shift

class BalPhaseCorrection (Operation):

    def run(self, image):
        if not image.ref_data or len(image.ref_vols) < 2:
            self.log("Not enough reference volumes, quitting.")
            return

        #find point-by-point the image-space angle between 2nd and 1st reference data, divide by two
        #fft & inverse_fft go row-by-row on arrays, so a one-liner:
        ref_phs = angle(conjugate(inverse_fft(image.ref_data[0]))*inverse_fft(image.ref_data[1]))/2

        #define the correction to subtract this angle
        correction = cos(ref_phs) - 1.0j*sin(ref_phs)

        #apply the correction in image space, take back to k space (correction volume
        #gets repeatedly multiplied to any volume in image.data)
        image.data[:] = fft(inverse_fft(image.data)*correction).astype(Complex32)

        #shift(image.data, 0, image.data.shape[-1]/2)
