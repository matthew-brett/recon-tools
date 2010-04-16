"""
This module finds the plane-rotation necessary to put the image into one
of the six ANALYZE-defined orientations (see the analyze.py module for info).
Alternatively, the target recon_epi can be used, which always makes the
same rotation and matches the output of the old recon_epi tool.

If the image is in a different plane than the orientation target, a warning
will arise and the image will not change.
"""

import numpy as np
from recon.operations import Operation, Parameter, ChannelIndependentOperation,\
     ChannelAwareOperation
from recon.imageio import ReconImage
from recon.analyze import xforms, canonical_orient
from recon import util, loads_extension_on_call

##############################################################################
class RotPlane (Operation):
    """
    The orientations in the operation are taken from ANALYZE orient codes
    and are left-handed. However, if the final image is to be NIFTI type,
    the rotation transform is updated (in the right-handed system).
    """

    params = (Parameter(name="orient_target", type="str", default=None,
                        description=
    """
    Final orientation of the image, taken from ANALYZE orient codes.
    Can be: radiological, transverse, coronal, coronal_flipped, sagittal, and
    sagittal_flipped. Also may be recon_epi."""),
              Parameter(name="force", type="bool", default=False,
                        description=
    """
    If your image is scanned even slightly off axis from X, Y, or Z in scanner
    space, then the transformation will not proceed. You can, however, force
    the rotation, and imply that the new orient target is the true mapping"""),)

    #@ChannelIndependentOperation
    @ChannelAwareOperation
    def run(self, image):

        if (self.orient_target not in xforms.keys() + ["recon_epi",]):
            self.log("no xform available for %s"%self.orient_target)
            return
        if self.orient_target == "recon_epi":
            # always swap -x to +y, and -y to +x
            Ts = np.array([[ 0.,-1., 0.],
                           [-1., 0., 0.],
                           [ 0., 0., 1.],])
            dest_xform = np.dot(image.orientation_xform.tomatrix(),
                                np.linalg.inv(Ts))
        else:
            dest_xform = xforms.get(self.orient_target, None)
            
        image.transform(new_mapping = dest_xform, force = self.force)
        
