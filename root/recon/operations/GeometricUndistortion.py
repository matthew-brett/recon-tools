import numpy as N

from recon.operations import Operation, Parameter, verify_scanner_image, \
     ChannelIndependentOperation
from recon.nifti import readImage
from recon.util import resample_phase_axis

##############################################################################
class GeometricUndistortion (Operation):
    """
    Performs a sub-pixel shifting to correct for susceptibility induced
    distortions in the PE direction
    """

    params = (
	Parameter(name="fmap_file", type="str", default="",
		  description="The name of the field map file."),
	)

    @ChannelIndependentOperation
    def run(self, image):
        "Correct for Nyquist ghosting due to field inhomogeneity."
        if not verify_scanner_image(self, image): return
	if not self.fmap_file:
	    self.log("No field map file provided, quitting")
	    return
        if not image.isepi:
            self.log("Can't apply inhomogeneity correction"\
                     " to non-epi sequences, quitting")
            return
	fMap = readImage(self.fmap_file)
        # grab the fieldmap, ignore the mask
        fMap = fMap.subImage(0)
	if fMap.shape[-3:] != image.shape[-3:]:
	    self.log("This field map's shape does not match"\
                     " the image shape, quitting")
	    return
	
	shift = (image.idim * image.T_pe/2/N.pi)
        
        #watch the sign
        pixel_pos = -shift*fMap[:] + N.outer(N.arange(fMap.jdim),
                                             N.ones(fMap.jdim))
        
        image[:].real = resample_phase_axis(abs(image[:]), pixel_pos)
        image[:].imag = 0.
	
        
