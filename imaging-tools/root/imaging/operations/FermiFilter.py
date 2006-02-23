from pylab import frange, meshgrid, exp, sqrt, Float32
from imaging.operations import Operation, Parameter
from imaging.util import fermi_filter


##############################################################################
class FermiFilter (Operation):
    "Apply a Fermi filter to the image."

    params=(
      Parameter(name="cutoff", type="float", default=0.95,
        description="distance from the center at which the filter drops to "\
        "0.5.  Units for cutoff are percentage of radius."),
      Parameter(name="trans_width", type="float", default=0.3,
        description="transition width.  Smaller values will result in a "\
        "sharper dropoff."))

    #-------------------------------------------------------------------------
    def run(self, image):
        rows, cols = image.data.shape[-2:]
        image.data *= fermi_filter(
          rows, cols, self.cutoff, self.trans_width).astype(Float32)


