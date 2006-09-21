from sliceview import sliceview
from recon.operations import Operation

##############################################################################
class ViewImage (Operation):
    "Run the sliceview volume viewer."
    #-------------------------------------------------------------------------
    def run(self, image):
        sliceview(image.data, ("Time Point", "Slice", "Row", "Column"))
