import sys
from imaging.varian.FDFImage import FDFImage
from imaging.operations.ReorderSlices import ReorderSlices

##############################################################################
class FDF2Img (object):

    #-------------------------------------------------------------------------
    def run(self):
        datadir = sys.argv[1]
        outputdir = datadir.split(".")[0]+".img"
        fdfimage = FDFImage(datadir)
        ReorderSlices().run(fdfimage)
        fdfimage.save(outputdir, "analyze")
