import sys
from os.path import join, basename, dirname, split
from optparse import OptionParser, Option
from recon.tools import ConsoleTool
from recon.imageio import readImage
from recon.operations.ViewImage import ViewImage
from recon.operations.ViewOrtho import ViewOrtho

##############################################################################
class ViewImageTool (ConsoleTool):
    "View the specified image in the sliceviewer."

    usage= "usage: %prog [options] image\n ", __doc__

    #-------------------------------------------------------------------------

    options = (
        Option("-f", "--file-format", dest="file_format", action="store",
               type="choice", default='analyze',
               choices=('analyze', 'nifti'), help="""input file type (can be
               analyze or nifti"""),
        )

    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self.add_options(self.options)
    
    def run(self, plotter="slice"):
        opts, args = self.parse_args()
        if not args:
            self.print_help()
            sys.exit(0)
        filetype = opts.file_format
        image = readImage(args[0], filetype)
        if plotter is "slice":
            ViewImage(title=args[0]).run(image)
        else:
            ViewOrtho(title=args[0]).run(image)
