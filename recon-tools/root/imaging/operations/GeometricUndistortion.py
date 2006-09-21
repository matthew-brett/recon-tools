from imaging.operations import Operation, Parameter
from imaging.analyze import readImage
from imaging.util import resample_phase_axis
from pylab import pi, arange, outerproduct, ones


##############################################################################
class GeometricUndistortion (Operation):

    params = (
	Parameter(name="fmap_file", type="str", default="",
		  description="The name of the field map file."),
	)

    def run(self, image):
        "Correct for Nyquist ghosting due to field inhomogeneity."
	
	if not self.fmap_file:
	    self.log("No field map file provided, quitting")
	    return
        if not image.isepi:
            self.log("Can't apply inhomogeneity correction"\
                     " to non-epi sequences, quitting")
            return
	fMap = readImage(self.fmap_file)
	if fMap.data.shape[-3:] != image.data.shape[-3:]:
	    self.log("This field map's shape does not match"\
                     " the image shape, quitting")
	    return
	
	shift = (image.xdim * image.T_pe/2/pi)
        
        #watch the sign
        pixel_pos = -shift*fMap.data + \
                    outerproduct(arange(fMap.ydim), ones(fMap.ydim))
        
        image.data.real = resample_phase_axis(abs(image.data), pixel_pos)
        image.data.imag = 0.
	
        
