# needs to be re-written

import numpy as N
from recon.operations import Operation, Parameter
from recon.util import Quaternion
from recon.analyze import canonical_orient

##############################################################################
class FlipSlices (Operation):
    """
    Flip image slices up-down and left-right
    """

    params=(
      Parameter(name="flipud", type="bool", default=False,
        description="flip each slice up-down"),
      Parameter(name="fliplr", type="bool", default=False,
        description="flip each slice left-right"))

    #-------------------------------------------------------------------------
    def run(self, image):

        if not self.flipud and not self.fliplr: return

        for vol in image:
            for sl in vol:
                if self.flipud and self.fliplr:
                    newslice = N.flipud(N.fliplr(sl[:]))
                elif self.flipud:
                    newslice = N.flipud(sl[:])
                elif self.fliplr:
                    newslice = N.fliplr(sl[:])
                sl[:] = newslice.copy()
        # book-keeping section
        new_xform = image.orientation_xform.tomatrix()
        if self.flipud:
            new_xform[:,1] = -new_xform[:,1]
        if self.fliplr:
            new_xform[:,0] = -new_xform[:,0]
        mapping = N.dot(N.linalg.inv(new_xform),
                        image.orientation_xform.tomatrix())
        new_orig = N.dot(mapping, N.array([image.x0, image.y0, image.z0]))
        (image.x0, image.y0, image.z0) = tuple(new_orig)
        image.orientation_xform = Quaternion(M=new_xform)
        image.orientation = canonical_orient(image.orientation_xform.tomatrix())
