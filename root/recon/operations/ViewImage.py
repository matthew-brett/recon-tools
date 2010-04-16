from recon.operations import Operation, Parameter, ChannelAwareOperation

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
    @ChannelAwareOperation
    def run(self, image):
        # make this raise any runtime error at actual runtime, instead
        # of load-time
        from recon.visualization.sliceview import sliceview
        dimnames = (image.tdim and ("Time Point",) or ()) + \
                   ("Slice", "Row", "Column",)
        sliceview(image, dimnames, title=self.title)
