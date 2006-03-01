"Applies a Balanced Phase Correction on data sets with two reference volumes"

from FFT import inverse_fft
from pylab import angle, conjugate, sin, cos, Complex32, fft, arange, reshape, ones, sqrt
from imaging.operations import Operation
from imaging.util import shift, ifft, apply_phase_correction

class BalPhaseCorrection (Operation):

    def run(self, image):
        if not image.ref_data or len(image.ref_vols) < 2:
            self.log("Not enough reference volumes, quitting.")
            return

        #find point-by-point the image-space angle between 2nd and 1st reference data, divide by two
        #fft & inverse_fft go row-by-row on arrays, so a one-liner:
        neg = ones((image.xdim, 1)) - 2*reshape((arange(image.xdim))%2, (image.xdim, 1))
        ref_phs = neg*angle(ifft(image.ref_data[1])*conjugate(ifft(image.ref_data[0])))/2

        #apply the correction in image space, take back to k space (correction volume
        #gets repeatedly multiplied to any volume in image.data)
        correction = cos(ref_phs) + 1.0j*sin(ref_phs)
        image.data[:] = fft(ifft(image.data)*correction).astype(Complex32)
        
        #image.data = apply_phase_correction(image.data, -ref_phs, shift=True)
        
        #shift(image.data, 0, image.data.shape[-1]/2)
