from Numeric import empty
from FFT import inverse_fft
from pylab import mlab, angle, fft, cos, sin, Float, Complex32
from imaging.operations import Operation


##############################################################################
class PhaseCorrection (Operation):

    #-------------------------------------------------------------------------
    def run(self, options, data):
        imgdata = data.data_matrix

        # phase angle of inverse fft'd reference volume
        ref_phs = angle(inverse_fft(data.ref_data))

        # compute phase correction
        cor = cos(-ref_phs) + 1.j*sin(-ref_phs)

        # apply correction to image data
        data.data_matrix = fft(inverse_fft(imgdata)*cor).astype(Complex32)
