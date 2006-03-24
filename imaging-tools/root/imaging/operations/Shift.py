from imaging.operations import Operation, Parameter
from imaging.util import shift

class Shift (Operation):
    "Allows a user to make arbitrary shifts of the data"
    params=(
        Parameter(name="yshift", type="int", default=0,
                  description="number of points to shift up and down"),
        Parameter(name="xshift", type="int", default=0,
                  description="number of points to shift left to right"))

    def run(self, image):
        shift(image.data, 0, self.xshift)
        shift(image.data, 1, self.yshift)

