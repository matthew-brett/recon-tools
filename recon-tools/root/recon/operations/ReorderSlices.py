from pylab import take, array, find
from recon.operations import Operation, Parameter, verify_scanner_image

##############################################################################
class ReorderSlices (Operation):
    "Reorder image slices from inferior to superior."
    #-------------------------------------------------------------------------
    def run(self, image):
        if not verify_scanner_image(self, image): return
        S = image.nslice
        acq_order = image.acq_order
        s_ind = array([find(acq_order==s)[0] for s in range(S)])
        image[:] = take(image[:], s_ind, axis=-3)
        if hasattr(image, 'ref_data'):
            ref = image.ref_data
            ref[:] = take(ref, s_ind, axis=-3)

        return 
        
