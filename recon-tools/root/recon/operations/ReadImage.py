from recon.imageio import readImage, available_readers, recon_output2dtype
from recon.operations import Operation, Parameter    

##############################################################################
class ReadImage (Operation):
    "Read image from file."

    params=(
      Parameter(name="filename", type="str", default="image",
        description="""
    File name prefix for output (extension will be determined by
    the format)."""),
      Parameter(name="format", type="str", default=None,
        description="""
    File format to write image as."""),
      Parameter(name="datatype", type="str", default=None,
        description="""
    Load incoming data as this data type (default is raw data type;
    only complex32 is supported for FID loading). Available datatypes:
    %s"""%recon_output2dtype.keys()),
      Parameter(name="vrange", type="tuple", default=(),
        description="""
    Volume range over-ride"""))
    #-------------------------------------------------------------------------
    def run(self):        
        return readImage(self.filename, self.format,
                         datatype=self.datatype, vrange=self.vrange)
