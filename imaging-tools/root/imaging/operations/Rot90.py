"""
This module defines a rotation on all slices, putting them into standard
radiological format.
"""
from pylab import reshape, rot90
from imaging.operations import Operation


##############################################################################
class Rot90 (Operation):
    """
    This class operatates slice-by-slice to rotate the images so that they are
    in standard radiological format.
    """

    def run(self, image):
        data = image.data
        for vol in data:
            for slice in vol:
                rotated = rot90(slice)
                reshape(slice, rotated.shape)
                slice[:] = rotated.copy()
        image.setData(data)

        # swap x and y origin locations (...is this right?)
        image.y0, image.x0 = image.x0, image.y0

        # swap x and y dimension sizes
        image.ysize, image.xsize = image.xsize, image.ysize
