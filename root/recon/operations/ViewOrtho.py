from recon.operations import Operation, Parameter, ChannelAwareOperation

##############################################################################
class ViewOrtho (Operation):
    """
    Run the orthogonal plane viewer.
    """
    params = (
        Parameter(name="title", type="str", default="spmclone",
                  description="optional name for the spmclone window"),
        )
    #-------------------------------------------------------------------------
    @ChannelAwareOperation
    def run(self, image):
        # make this raise any runtime error at actual runtime, instead
        # of load-time
        from recon.visualization.spmclone import spmclone
        spmclone(image, title=self.title)
