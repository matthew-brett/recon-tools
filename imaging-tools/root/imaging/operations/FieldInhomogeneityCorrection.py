from imaging.operations import Operation, Parameter
from imaging.analyze import readImage
from imaging.util import resample_phase_axis
from pylab import pi, Float32, arange, squeeze


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
	fMap = readImage(self.fmap_file)
## 	if fMap.data.shape[-3:] != image.data.shape[-3:]:
## 	    self.log("This field map's shape does not match the image shape, quitting")
## 	    return
	
	shift = (image.xdim * image.dwell_time / (2*pi))
	pixel_pos = shift*fMap.data + arange(fMap.xdim)

        #for slice, pix_pos in (pic, pixel_pos):
        for slice in image.data:
            for z in range(fMap.zdim):
                slice[z,:] = resample_phase_axis(abs(slice[z]), pixel_pos[z])
	
        # Read the inhomogeneity field-map data from disk (Calculated using compute_fmap).
       
        # Loop over each phase-encode line of uncorrected data S'_mn.
           
            # Perform FFT of S'_mn with respect to n (frequency-encode direction) to
            # obtain S'_m(x).

            # Loop over each point x.

                # Calculate the perturbation kernel K_mm'(x) at x by: 
                # (1) Performing FFT with respect to y' of exp(i*2*pi*phi(x,y')*Delta t)
                #     and exp(i*2*pi*phi(x,y')*Delta t).
                # (2)  

                # Invert perturbation operator to obtain correction operator at x.

                # Apply correction operator at x to the distorted data S'_m(x) to
                # obtain the corrected data S_m(x).

            # Perform an inverse FFT in the read direction to obtain the corrected
            # k-space data S_mn.
