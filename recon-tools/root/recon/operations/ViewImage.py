from sliceview import sliceview
from recon.operations import Operation

##############################################################################
class ViewImage (Operation):
    "Run the sliceview volume viewer."
    #-------------------------------------------------------------------------
    def run(self, image):
        dimnames = (image.tdim and ("Time Point",) or ()) + \
                   ("Slice", "Row", "Column",)
        sliceview(image[:], dimnames)
