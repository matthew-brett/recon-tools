from imaging.operations import Operation, Parameter
from imaging.analyze import readImage
from imaging.util import resample_phase_axis
from pylab import pi, Float32, Complex32, arange, outerproduct, ones


##############################################################################
class FieldInhomogeneityCorrection (Operation):

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
	
	shift = (image.xdim * image.dwell_time / (2*pi))
        pixel_pos = shift*fMap.data.real + \
                    outerproduct(arange(fMap.xdim), ones(fMap.xdim))
        # will redo this loop after revisiting resample_phase_axis
        
        for vol in image.data:
            for z in range(fMap.zdim):
                vol[z,:] = resample_phase_axis(abs(vol[z]), pixel_pos[z]).astype(Complex32)
	
        
