from recon.imageio import readImage, available_readers
from recon.operations import Operation, Parameter
from recon.operations.WriteImage import param2dtype

    

##############################################################################
class ReadImage (Operation):
    "Read image from file."

    params=(
      Parameter(name="filename", type="str", default="image",
        description="File name prefix for output (extension will be "\
                    "determined by the format)."),
      Parameter(name="format", type="str", default="analyze",
        description="File format to write image in.  (Should eventually be"\
                    "auto-detected)"),
      Parameter(name="datatype", type="str", default=None,
        description="Load incoming data as this data type (default is raw "\
                "data type; only complex32 is supported for FID loading). " \
                "Available datatypes: %s"%param2dtype.keys()),
      Parameter(name="vrange", type="tuple", default=(),
        description="Volume range over-ride"))
    #-------------------------------------------------------------------------
    def run(self):
        if self.format not in available_readers:
            print "Unsupported input type: %s"%self.format
            return

        new_dtype = self.datatype and \
                    param2dtype.get(self.datatype, False) or None
        if new_dtype is False:
            raise ValueError("unsupported data type: %s"%self.datatype)
        
        return readImage(self.filename, self.format,
                         target_dtype=new_dtype, vrange=self.vrange)
