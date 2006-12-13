from pylab import flipud, fliplr
from recon.operations import Operation, Parameter
from recon.util import Quaternion

##############################################################################
class FlipSlices (Operation):
    "Flip image slices up-down and left-right"

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
                    newslice = flipud(fliplr(sl[:]))
                elif self.flipud:
                    newslice = flipud(sl[:])
                elif self.fliplr:
                    newslice = fliplr(sl[:])
                sl[:] = newslice.copy()
                
        mat = image.orientation_xform.tomatrix()
        if self.flipud:
            mat[:,1] = -mat[:,1]
        if self.fliplr:
            mat[:,0] = -mat[:,0]
        image.orientation_xform = Quaternion(M=mat)        
