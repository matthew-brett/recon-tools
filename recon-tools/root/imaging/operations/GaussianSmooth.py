from recon.operations import Operation, Parameter
from Numeric import exp, power, log, arange, outerproduct, pi
from scipy.signal import convolve

def gaussian_smooth(data, fwhm, kernSize):
    sigma = fwhm*power(8 * log(2), -0.5)
    gax = arange(kernSize) - (kernSize-1)/2.
    gf = exp(-power(gax,2)/(2*sigma**2))
    kern = outerproduct(gf,gf)/(2*pi*sigma**2)
    return convolve(kern, data, mode='same')

class GaussianSmooth (Operation):
    params = (
        Parameter(name="fwhm", type="float", default=2.0,
                  description="full width at half max of gaussian dist"),
        Parameter(name="kernel_rank", type="int", default=3,
                  description="rank of smoothing kernel (gaussian will be "\
                  "centered at a reasonable spot")
        )

    # at least it runs, make it less loopy later
    def run(self, image):
        for vol in image.data:
            for s in vol:
                s[:] = gaussian_smooth(s, self.fwhm, self.kernel_rank).astype(s.typecode())
        
    
