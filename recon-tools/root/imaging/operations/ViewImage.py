from sliceview import sliceview
from imaging.operations import Operation

##############################################################################
class ViewImage (Operation):
    "Run the sliceview volume viewer."
    #-------------------------------------------------------------------------
    def run(self, image):
        sliceview(image.data, ("Time Point", "Slice", "Row", "Column"))
