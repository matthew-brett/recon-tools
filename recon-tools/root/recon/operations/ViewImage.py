from recon.visualization.sliceview import sliceview
from recon.operations import Operation, Parameter

##############################################################################
class ViewImage (Operation):
    """
    Run the sliceview volume viewer.
    """
    params = (
        Parameter(name="title", type="str", default="sliceview",
                  description="optional name for the sliceview window"),
        )
    #-------------------------------------------------------------------------
    def run(self, image):
        dimnames = (image.tdim and ("Time Point",) or ()) + \
                   ("Slice", "Row", "Column",)
        sliceview(image, dimnames, title=self.title)
