import sys
from os.path import join, basename, dirname
from optparse import OptionParser

from imaging.imageio import readImage
from imaging.operations.ViewImage import ViewImage

##############################################################################
class ViewImageTool (object):

    # command line option parser
    parser = OptionParser(usage=\
      "  usage: %prog [options] image\n"\
      "  View the specified Analyze7.5 formatted image.")

    #-------------------------------------------------------------------------
    def run(self):
        _, args = self.parser.parse_args()
        if not args:
            self.parser.print_help()
            sys.exit(0)

        # remove extension if present
        filestem = join(dirname(args[0]), basename(args[0]).rsplit(".",1)[0])
        image = readImage(filestem, "analyze")
        image.info()
        ViewImage().run(image)
