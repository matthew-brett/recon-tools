from Numeric import empty, NewAxis
from FFT import inverse_fft
from pylab import mlab, pi, fft, floor, angle, where, amax, cos, sin, Float, Complex32
from recon.operations import Operation
from recon.util import nice_angle


##############################################################################
class SegmentationCorrection (Operation):
    """
    Correct for the Nyquist ghosting in segmented scans due to mismatches
    between segments.
    """
    
    #-------------------------------------------------------------------------
    def run(self, options, data):
        imgdata = data.data_matrix
        pe_per_seg = data.n_pe_true/data.nseg

        # phase angle of inverse fft'd ref navs and image navs
        ref_nav_phs = angle(inverse_fft(data.ref_nav_data))
        nav_phs = angle(inverse_fft(data.nav_data))

        # phase difference between ref navs and image navs
        phsdiff = nice_angle(ref_nav_phs - nav_phs)

        # weight phase difference by the phase encode timing during each segment
        pe_times = (data.pe_times[data.nav_per_seg:]/data.echo_time)[:,NewAxis]
        theta = empty(imgdata.shape, Float)
        theta[:,:,:pe_per_seg] = phsdiff[:,:,NewAxis,0]*pe_times
        theta[:,:,pe_per_seg:] = phsdiff[:,:,NewAxis,1]*pe_times

        # Compute the phase correction.
        cor = cos(theta) + 1.0j*sin(theta)

        # Apply the phase correction.
        data.data_matrix = fft(inverse_fft(imgdata)*cor).astype(Complex32)
                            


