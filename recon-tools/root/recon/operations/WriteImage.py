import Numeric as N
from recon.operations import Operation, Parameter
from recon.util import import_from, integer_ranges, scale_data
from recon.imageio import writeImage
from odict import odict

param2dtype = odict((
    ('magnitude', N.Float32),
    ('compmlex', N.Complex32),
    ('double', N.Float),
    ('byte', N.Int8),
    ('short', N.Int16),
    ('int', N.Int32),
    ))

# ReconTools default = idx 0
output_datatypes = param2dtype.keys()

class WriteImage (Operation):
    "Write an image to the filesystem."

    params=(
      Parameter(name="filename", type="str", default="image",
        description="File name prefix for output (extension will be "\
                    "determined by the format)."),
      Parameter(name="suffix", type="str", default=None,
        description="Over-rides the default suffix behavior."),
      Parameter(name="filedim", type="int", default=3,
        description="Number of dimensions per output file."),
      Parameter(name="format", type="str", default="analyze",
        description="File format to write image in."),
      Parameter(name="datatype", type="str", default="magnitude",
        description="Output datatype options: %s.\nDefault is"\
                    "magnitude."%output_datatypes))

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

        data_code = writer.typecode2datatype[new_dtype]
        # do a sanity check
        if not hasattr(image[:], "imag") and data_code == writer.COMPLEX:
            data_code = writer.FLOAT
        
        niftitype = self.format.find("nifti") > -1 and self.format[6:] or None
        writeImage(image, self.filename, datatype=data_code, filetype=niftitype,
                   targetdim=self.filedim, suffix=self.suffix,
                   scale=scale, format=self.format)
