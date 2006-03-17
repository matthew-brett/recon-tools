from imaging.imageio import readImage, available_readers
from imaging.operations import Operation, Parameter


##############################################################################
class ReadImage (Operation):
    "Read image from file."

    params=(
      Parameter(name="filename", type="str", default="image",
        description="File name prefix for output (extension will be "\
                    "determined by the format)."),
      Parameter(name="format", type="str", default="analyze",
        description="File format to write image in.  (Should eventually be"\
                    "auto-detected)"))

    #-------------------------------------------------------------------------
    def run(self):
        if self.format not in available_readers:
            print "Unsupported input type: %s"%self.format
            return
        return readImage(self.filename, self.format)