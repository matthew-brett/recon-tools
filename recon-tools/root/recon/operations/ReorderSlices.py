from Numeric import empty
from pylab import mlab, zeros, arange, take, array, find
from recon.operations import Operation, Parameter
from recon.util import reverse

##############################################################################
class ReorderSlices (Operation):
    "Reorder image slices from inferior to superior."

    params=(
      Parameter(name="flip_slices", type="bool", default=False,
        description="Flip slices during reordering."),)

    #-------------------------------------------------------------------------
    def run(self, image):
        S = image.nslice
        img = image.data
        acq_order = array(range(S-1,-1,-2) + range(S-2,-1,-2))
        s_ind = array([find(acq_order==s)[0] for s in range(S)])
        img[:] = self.flip_slices and take(img, reverse(s_ind), axis=1) \
                                  or take(img, s_ind, axis=1)
        if hasattr(image, 'ref_data'):
            ref = image.ref_data
            ref[:] = self.flip_slices and take(ref, reverse(s_ind), axis=1) \
                                      or take(ref, s_ind, axis=1)

        return 
        
