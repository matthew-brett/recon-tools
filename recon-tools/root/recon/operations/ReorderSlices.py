from Numeric import empty
from pylab import mlab, zeros, arange, take
from recon.operations import Operation, Parameter

##############################################################################
class ReorderSlices (Operation):
    "Reorder image slices from inferior to superior."

    params=(
      Parameter(name="flip_slices", type="bool", default=False,
        description="Flip slices during reordering."),)

    #-------------------------------------------------------------------------
    def run(self, options, data):
        nslice = data.nslice
        imgdata = data.data_matrix
        # this needs testing on odd number of slices
        midpoint = nslice/2 + (nslice%2 and 1 or 0)
        tmp = empty(imgdata.shape, imgdata.typecode())
        indices = nslice - 1 - arange(nslice)
        tmp[:,::2] = take(imgdata, indices[:midpoint], axis=1)
        tmp[:,1::2] = take(imgdata, indices[midpoint:], axis=1)
        imgdata[:] = self.flip_slices and take(tmp, indices, axis=1) or tmp
