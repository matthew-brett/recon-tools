from pylab import flipud, fliplr
from imaging.operations import Operation, Parameter

##############################################################################
class FlipSlices (Operation):
    "Flip image slices up-down and left-right"

    params=(
      Parameter(name="flipud", type="bool", default=True,
        description="flip each slice up-down"),
      Parameter(name="fliplr", type="bool", default=True,
        description="flip each slice left-right"))

    #-------------------------------------------------------------------------
    def run(self, image):
        if not self.flipud and not self.fliplr: return
        for volume in image.data:
            for slice in volume:
                if self.flipud and self.fliplr:
                    newslice = flipud(fliplr(slice))
                elif self.flipud:
                    newslice = flipud(slice)
                elif self.fliplr:
                    newslice = fliplr(slice)
                slice[:] = newslice
