from Numeric import empty
from pylab import mlab, zeros, arange, take
from imaging.operations import Operation, Parameter

##############################################################################
class ReorderSlices (Operation):
    "Reorder image slices from inferior to superior."

    params=(
      Parameter(name="flip_slices", type="bool", default=False,
        description="Flip slices during reordering."),)

    #-------------------------------------------------------------------------
    def run(self, image):
        nslice = image.nslice
        imgdata = image.data
        refdata = image.ref_data
        # this needs testing on odd number of slices
        midpoint = nslice/2 + (nslice%2 and 1 or 0)
        tmp = empty(imgdata.shape, imgdata.typecode())
        indices = nslice - 1 - arange(nslice)
        tmp[:,::2] = take(imgdata, indices[:midpoint], axis=1)
        tmp[:,1::2] = take(imgdata, indices[midpoint:], axis=1)
        imgdata[:] = self.flip_slices and take(tmp, indices, axis=1) or tmp
        tmp = empty(refdata.shape, refdata.typecode())
        tmp[:,::2] = take(refdata, indices[:midpoint], axis=1)
        tmp[:,1::2] = take(refdata, indices[midpoint:], axis=1)
        refdata[:] = self.flip_slices and take(tmp, indices, axis=1) or tmp
        
