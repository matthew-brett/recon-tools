"""
This module defines a rotation on all slices, putting them into standard
radiological format.
"""
from pylab import reshape, rot90, flipud
from imaging.operations import Operation
from imaging.imageio import BaseImage

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
                #rotated = flipud(rot90(slice))
                rotated = rot90(slice)
                reshape(slice, rotated.shape)
                slice[:] = rotated.copy()
        image.setData(data)
        image.noteRot()
