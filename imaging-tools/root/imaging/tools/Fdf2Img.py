import os
import sys
from imaging.imageio import write_analyze, AnalyzeWriter
from imaging.varian.FDFImage import FDFImage
from imaging.operations.ReorderSlices import ReorderSlices

##############################################################################
class Fdf2Img (object):

    #-------------------------------------------------------------------------
    def run(self):
        datadir = sys.argv[1]
        outputdir = datadir.split(".")[0]+".img"

        # make sure the output directory is in place
        if not os.path.exists(outputdir): os.mkdir(outputdir)

        fdfimage = FDFImage(datadir)
        ReorderSlices().run(None, fdfimage)
        fdfimage.save(outputdir)
