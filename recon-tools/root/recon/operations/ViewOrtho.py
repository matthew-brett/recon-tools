from recon.visualization.spmclone import spmclone
from recon.operations import Operation, Parameter

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
    def run(self, image):
        spmclone(image, title=self.title)
