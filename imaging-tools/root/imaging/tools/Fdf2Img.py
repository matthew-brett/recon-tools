import os
import sys
from optparse import OptionParser

from imaging.varian.FDFImage import FDFImage
from imaging.operations.ReorderSlices import ReorderSlices
from imaging.operations.FlipSlices import FlipSlices

##############################################################################
class Fdf2Img (object):

    # command line option parser
    parser = OptionParser(usage=\
      "  usage: %prog [options] fdf_directory\n"\
      "  Convert the the Varian FDF image in the given fdf_directory\n"\
      "  (.dat) into an Analyze image.  The Analyze dataset will be\n"\
      "  placed in a new directory with the same name but a .img extension\n"\
      "  instead of .dat.")

    #-------------------------------------------------------------------------
    def run(self):
        _, args = self.parser.parse_args()

        if not args:
            self.parser.print_help()
            sys.exit(0)
        else: datadir = args[0]
        outputdir = datadir.rsplit(".",1)[0]+".img"

        # make sure the output directory is in place
        if not os.path.exists(outputdir): os.mkdir(outputdir)

        fdfimage = FDFImage(datadir)
        ReorderSlices().run(fdfimage)
        FlipSlices(flipud=False, fliplr=True).run(fdfimage)
        fdfimage.save(outputdir)
