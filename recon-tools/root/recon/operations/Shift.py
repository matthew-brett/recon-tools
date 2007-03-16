from recon.operations import Operation, Parameter
from recon.util import shift

class Shift (Operation):
    """
    Allows a user to make arbitrary shifts of the data.
    """
    params=(
        Parameter(name="yshift", type="int", default=0,
                  description="number of points to shift up and down"),
        Parameter(name="xshift", type="int", default=0,
                  description="number of points to shift left to right"))

    def run(self, image):
        if self.xshift:
            shift(image[:], 0, self.xshift)
        if self.yshift:
            shift(image[:], 1, self.yshift)

