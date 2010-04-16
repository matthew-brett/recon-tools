import numpy as np

from recon.operations import Operation, Parameter, ChannelIndependentOperation
from recon.util import fftconvolve
#from scipy.signal import fftconvolve

def gaussian_smooth(arr, sy, sx, kern_dims):
    """
    Smooth a N-dimensional matrix along the last two dimensions
    with a kernSize^2 gaussian kernel
    @param fwhm: full width at half max of gaussian distribution
    @param kernel_rank: rank of the smoothing kernel
    @return: smoothed matrix
    """
    ndim = len(arr.shape)
    m,n = kern_dims
    if ndim < 2:
        raise ValueError("Matrix dimension must be at least 2")
    xax = np.arange(-(n/2),n/2+1)
    yax = np.arange(-(m/2),m/2+1)
    
    kern = np.exp(-(yax[:,None]**2/(2*sy**2) + xax[None,:]**2/(2*sx**2)))
    kern /= (2*np.pi*sy*sx)
    # use newaxis to add dummy-dimensions to kern
    slicer = (np.newaxis,)*(ndim-2) + (slice(None),)*2
    return fftconvolve(arr, kern[slicer], mode='same', axes=(-2,-1))

class GaussianSmooth (Operation):
    """
    Apply a gaussian kernel to the image. Specify FWHM and kernel rank.
    """

    params = (
        Parameter(name='fwhm', type='float', default=2.0,
                  description="""
    Full width at half max of gaussian dist in mm"""),
        Parameter(name='rank_factor', type='int', default=3,
                  description="""
    Make the rank of the smoothing kernel appx this many times the stdev{x,y}"""
                  ),
        Parameter(name='SOS', type='bool', default=True,
                  description="""
    Act only on the sum-of-squares combined image if possible."""
                  ),
        )
    
    @ChannelIndependentOperation
    def run(self, image):
        fwhm_x_pix = self.fwhm/image.isize
        fwhm_y_pix = self.fwhm/image.jsize
        fwhm_scale = (8*np.log(2))**0.5
        sx = fwhm_x_pix/fwhm_scale; sy = fwhm_y_pix/fwhm_scale
        scale = self.rank_factor
        M = 2*int(scale*sy + 0.99); N = 2*int(scale*sx + 0.99)
        image[:] = gaussian_smooth(image[:], sy, sx, (M,N))
        
    
