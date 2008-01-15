import numpy as N
from recon.operations import Operation, Parameter, verify_scanner_image

##############################################################################
class ReorderSlices (Operation):
    """
    Reorder image slices from acquisition order to physical order.
    """
    #-------------------------------------------------------------------------
    def run(self, image):
        if not verify_scanner_image(self, image): return
        S = image.n_slice
        acq_order = image.acq_order
        s_ind = N.concatenate([N.nonzero(acq_order==s)[0] for s in range(S)])
        image[:] = N.take(image[:], s_ind, axis=-3)
        if hasattr(image, 'ref_data'):
            ref = image.ref_data
            ref[:] = N.take(ref, s_ind, axis=-3)

        return 
        
