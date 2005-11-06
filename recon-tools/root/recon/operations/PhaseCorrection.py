from Numeric import empty
from FFT import inverse_fft
from pylab import mlab, angle, fft, cos, sin, Float, Complex32
from recon.operations import Operation


##############################################################################
class PhaseCorrection (Operation):

    #-------------------------------------------------------------------------
    def run(self, options, data):
        ksp_data = data.data_matrix

        # phase angle of inverse fft'd reference volume
        ref_phs = angle(inverse_fft(data.ref_data))

        # Apply the phase correction to the image data.
        for volume in ksp_data:
            for slice, theta in zip(volume, ref_phs):
                cor = cos(-theta) + 1.0j*sin(-theta)
                slice[:] = fft(inverse_fft(slice)*cor).astype(Complex32)
