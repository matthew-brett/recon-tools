from pylab import frange, meshgrid, exp, sqrt, Float32
from imaging.operations import Operation, Parameter


##############################################################################
def fermi_filter(rows, cols, cutoff, trans_width):
    """
    @return a Fermi filter kernel.
    @param cutoff: distance from the center at which the filter drops to 0.5.
      Units for cutoff are percentage of radius.
    @param trans_width: width of the transition.  Smaller values will result
      in a sharper dropoff.
    """
    row_end = (rows-1)/2.0; col_end = (cols-1)/2.0
    row_vals = frange(-row_end, row_end)**2
    col_vals = frange(-col_end, col_end)**2
    X, Y = meshgrid(row_vals, col_vals)
    return 1/(1 + exp((sqrt(X + Y) - cutoff*cols/2.0)/trans_width))


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
        kernel = fermi_filter(
          rows, cols, self.cutoff, self.trans_width).astype(Float32)
        for volume in image.data:
          for slice in volume: slice *= kernel


