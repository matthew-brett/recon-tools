import numpy as N
from recon.operations import Operation, Parameter
from recon.imageio import output_datatypes

class WriteImage (Operation):
    """
    Write an image to the filesystem.
    """

    params=(
      Parameter(name="filename", type="str", default="image",
        description="""
    File name prefix for output (extension is
    determined by the format)."""),
      Parameter(name="suffix", type="str", default=None,
        description="""
    Over-rides the default suffix behavior."""),
      Parameter(name="filedim", type="int", default=3,
        description="""
    Number of dimensions per output file."""),
      Parameter(name="format", type="str", default="analyze",
        description="""
    File format to write image as."""),
      Parameter(name="datatype", type="str", default="magnitude",
        description="""
    Output datatype options: %s."""%output_datatypes))

    def run(self, image):
        image.writeImage(self.filename, format_type=self.format,
                         datatype=self.datatype, targetdim=self.filedim,
                         suffix=self.suffix)
        
