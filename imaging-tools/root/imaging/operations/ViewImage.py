from sliceview import sliceview
from imaging.operations import Operation, Parameter

##############################################################################
class ViewImage (Operation):
    "Flip image slices up-down and left-right"
    #-------------------------------------------------------------------------
    def run(self, image):
        sliceview(image.data, ("Time Point", "Slice", "Row", "Column"))
