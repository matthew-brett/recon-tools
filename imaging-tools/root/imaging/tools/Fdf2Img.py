import os
import sys
from imaging.imageio import write_analyze, AnalyzeWriter
from imaging.varian.FDFImage import FDFImage
from imaging.operations.ReorderSlices import ReorderSlices
from imaging.operations.FlipSlices import FlipSlices
from imaging.operations.ViewImage import ViewImage

##############################################################################
class Fdf2Img (object):

    #-------------------------------------------------------------------------
    def run(self):
        datadir = sys.argv[1]
        outputdir = datadir.rsplit(".",1)[0]+".img"

        # make sure the output directory is in place
        if not os.path.exists(outputdir): os.mkdir(outputdir)

        fdfimage = FDFImage(datadir)
        ReorderSlices().run(fdfimage)
        FlipSlices(flipud=False, fliplr=True).run(fdfimage)
        ViewImage().run(fdfimage)
        fdfimage.save(outputdir)
