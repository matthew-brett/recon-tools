from recon.operations import Operation, Parameter
from Numeric import exp, power, log, arange, outerproduct, pi, asarray, NewAxis
from recon.util import fftconvolve

def gaussian_smooth(M, fwhm, kernSize):
    """ Smooth a N-dimensional matrix along the last two dimensions
    with a kernSize^2 gaussian kernel
    @param fwhm: full width at half max of gaussian distribution
    @param kernel_rank: rank of the smoothing kernel
    @return: smoothed matrix
    """
    ndim = len(M.shape)
    if ndim < 2:
        raise ValueError("Matrix dimension must be at least 2")
    # set up guassian kernel
    sigma = fwhm*power(8 * log(2), -0.5)
    gax = arange(kernSize) - (kernSize-1)/2.
    gf = exp(-power(gax,2)/(2*sigma**2))
    kern = outerproduct(gf,gf)/(2*pi*sigma**2)
    # use NewAxis to add dummy-dimensions to kern
    slicer = (NewAxis,)*(ndim-2) + (slice(0,kernSize),)*2
    return fftconvolve(M, kern[slicer], mode='same', axes=(-2,-1))

class GaussianSmooth (Operation):
    """
    Apply a gaussian kernel to the image. Specify FWHM and kernel rank.
    """

    params = (
        Parameter(name="fwhm", type="float", default=2.0,
                  description="full width at half max of gaussian dist"),
        Parameter(name="kernel_rank", type="int", default=3,
                  description="rank of smoothing kernel (gaussian will be "\
                  "centered at a reasonable spot")
        )
    
    def run(self, image):
        image[:] = gaussian_smooth(image.data, self.fwhm,
                                   self.kernel_rank).astype(image.data.typecode())
        
    
