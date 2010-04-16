import numpy as N
from recon.operations import Operation, Parameter, ChannelIndependentOperation
##############################################################################
class FlipSlices (Operation):
    """
    Flip image slices up-down and left-right (in voxel space, not real space)
    """

    params=(
      Parameter(name="flipud", type="bool", default=False,
        description="flip each slice up-down"),
      Parameter(name="fliplr", type="bool", default=False,
        description="flip each slice left-right"))

    #-------------------------------------------------------------------------
    @ChannelIndependentOperation
    def run(self, image):

        if not self.flipud and not self.fliplr: return
        new_xform = image.orientation_xform.tomatrix()
        if self.fliplr:
            new_xform[:,0] = -new_xform[:,0]
        if self.flipud:
            new_xform[:,1] = -new_xform[:,1]
        image.transform(new_mapping=new_xform)
            
