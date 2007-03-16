import numpy as N
from recon.operations import Operation, Parameter
from recon.util import import_from, integer_ranges, scale_data
from odict import odict

param2dtype = odict((
    ('magnitude', N.dtype(N.float32)),
    ('complex', N.dtype(N.complex64)),
    ('double', N.dtype(N.float64)),
    ('byte', N.dtype(N.int8)),
    ('ubyte', N.dtype(N.uint8)),
    ('short', N.dtype(N.int16)),
    ('ushort', N.dtype(N.uint16)),
    ('int', N.dtype(N.int32)),
    ('uint', N.dtype(N.uint32)),
    ))

# ReconTools default = idx 0
output_datatypes = param2dtype.keys()

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
        # load the module that owns the Writer
        writer = {
            "analyze": "analyze",
            "nifti-single": "nifti",
            "nifti-dual": "nifti",
            }.get(self.format, None)
        if writer is None:
            raise ValueError("Unsupported output type: %s"%self.format)
        writer = import_from("recon", writer)

        # find out the new dtype to write the file with
        new_dtype = param2dtype.get(self.datatype.lower(), None)
        if new_dtype is None:
            raise ValueError("Unsupported data type: %s"%self.datatype)

        # check if the data needs a scale factor (maybe intercept)
        if new_dtype in integer_ranges.keys():
            scale = float(scale_data(image[:], new_dtype))
        else:
            scale = float(1.0)

        data_code = writer.dtype2datatype.get(new_dtype, None)
        if data_code is None:
            raise ValueError("Chosen data type is not supported by %s"%writer.__name__)
        # do a sanity check
        if not hasattr(image[:], "imag") and data_code == writer.COMPLEX:
            data_code = writer.FLOAT
        
        image.writeImage(self.filename, format_type=self.format,
                         datatype=data_code,targetdim=self.filedim,
                         suffix=self.suffix, scale=scale)
