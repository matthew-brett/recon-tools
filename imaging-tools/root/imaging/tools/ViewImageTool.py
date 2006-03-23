import sys
from os.path import join, basename, dirname

from imaging.tools import ConsoleTool
from imaging.imageio import readImage
from imaging.operations.ViewImage import ViewImage

##############################################################################
class ViewImageTool (ConsoleTool):
    "View the specified Analyze7.5 formatted image."

    usage= "usage: %prog [options] image\n ", __doc__

    #-------------------------------------------------------------------------
    def run(self):
        _, args = self.parse_args()
        if not args:
            self.print_help()
            sys.exit(0)

        # remove extension if present
        filestem = join(dirname(args[0]), basename(args[0]).rsplit(".",1)[0])
        image = readImage(filestem, "analyze")
        image.info()
        ViewImage().run(image)
