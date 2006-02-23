from Numeric import empty, NewAxis
from FFT import inverse_fft
from pylab import mlab, pi, fft, floor, angle, where, amax, cos, sin, Float, Complex32
from imaging.operations import Operation
from imaging.util import normalize_angle, shift, apply_phase_correction


##############################################################################
class SegmentationCorrection (Operation):
    """
    Correct for the Nyquist ghosting in segmented scans due to mismatches
    between segments.
    """
    
    #-------------------------------------------------------------------------
    def run(self, image):
        # can't perform segmentation correction on a non-segmented image!
        if image.nseg < 2:
            self.log("Image is non-segmented, nothing to do.")
            return
        pe_per_seg = image.n_pe_true/image.nseg

        # phase angle of inverse fft'd ref navs and image navs
        ref_nav_phs = angle(inverse_fft(image.ref_nav_data))
        nav_phs = angle(inverse_fft(image.nav_data))

        # phase difference between ref navs and image navs
        phsdiff = normalize_angle(ref_nav_phs - nav_phs)

        # weight phase difference by the phase encode timing during each segment
        pe_times = (image.pe_times[image.nav_per_seg:]/image.echo_time)[:,NewAxis]
        theta = empty(image.data.shape, Float)
        theta[:,:,:pe_per_seg] = phsdiff[:,:,NewAxis,0]*pe_times
        theta[:,:,pe_per_seg:] = phsdiff[:,:,NewAxis,1]*pe_times

        # Apply the phase correction.
        image.data = apply_phase_correction(image.data, theta)
