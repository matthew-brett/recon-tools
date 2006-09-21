import os
import sys

from recon.tools import ConsoleTool
from recon.varian.FDFImage import FDFImage
from recon.operations.ReorderSlices import ReorderSlices
from recon.operations.FlipSlices import FlipSlices

##############################################################################
class Fdf2Img (ConsoleTool):

    # command line usage
    usage=\
      "  usage: %prog [options] fdf_directory\n"\
      "  Convert the the Varian FDF image in the given fdf_directory\n"\
      "  (.dat) into an Analyze image.  The Analyze dataset will be\n"\
      "  placed in a new directory with the same name but a .img extension\n"\
      "  instead of .dat."

    #-------------------------------------------------------------------------
    def run(self):
        _, args = self.parse_args()

        if not args:
            self.print_help()
            sys.exit(0)
        else: datadir = args[0]
        outputdir = datadir.rsplit(".",1)[0]+".img"

        # make sure the output directory is in place
        if not os.path.exists(outputdir): os.mkdir(outputdir)

        fdfimage = FDFImage(datadir)
        ReorderSlices().run(fdfimage)
        FlipSlices(flipud=False, fliplr=True).run(fdfimage)
        fdfimage.save(outputdir)
