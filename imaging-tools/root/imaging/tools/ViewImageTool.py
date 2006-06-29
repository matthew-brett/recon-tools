import sys
from os.path import join, basename, dirname
from optparse import OptionParser, Option
from imaging.tools import ConsoleTool
from imaging.imageio import readImage
from imaging.operations.ViewImage import ViewImage

##############################################################################
class ViewImageTool (ConsoleTool):
    "View the specified Analyze7.5 formatted image."

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
    
    def run(self):
        opts, args = self.parse_args()
        if not args:
            self.print_help()
            sys.exit(0)
        filetype = opts.file_format
        # remove extension if present
        pruned_exts = ['nii', 'hdr', 'img']
        file_ext = basename(args[0]).rsplit(".",1)[1]
        filestem = file_ext in pruned_exts and \
                   join(dirname(args[0]), basename(args[0]).rsplit(".",1)[0]) \
                   or args[0]
        image = readImage(filestem, filetype)
        image.info()
        ViewImage().run(image)
