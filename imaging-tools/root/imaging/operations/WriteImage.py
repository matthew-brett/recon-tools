from imaging.imageio import writeImage
from imaging.operations import Operation, Parameter

ANALYZE_FORMAT = "analyze"
NIFTI_DUAL = "nifti dual"
NIFTI_SINGLE = "nifti single"
MAGNITUDE_TYPE = "magnitude"
COMPLEX_TYPE = "complex"

##############################################################################
class WriteImage (Operation):
    "Write image to the filesystem."

    params=(
      Parameter(name="filename", type="str", default="image",
        description="File name prefix for output (extension will be "\
                    "determined by the format)."),
      Parameter(name="format", type="str", default="analyze",
        description="File format to write image in."),
      Parameter(name="datatype", type="str", default="magnitude",
        description="Output datatype (complex or magnitude). Default is "\
                    "magnitude."))

    #-------------------------------------------------------------------------
    def writeAnalyze(self, image):
        from imaging import analyze

        # convert to format-specific datatype constant
        data_type = {
          MAGNITUDE_TYPE: analyze.FLOAT,
          COMPLEX_TYPE: analyze.COMPLEX
        }[self.datatype]
        analyze.writeImage(image, self.filename, data_type, 3)

    #-------------------------------------------------------------------------
    def writeNifti(self, image):
        from imaging import nifti

        # convert to format-specific datatype constant
        data_type = {
          MAGNITUDE_TYPE: nifti.FLOAT,
          COMPLEX_TYPE: nifti.COMPLEX
        }[self.datatype]
        nifti.writeImage(image, self.filename, data_type, 3, self.format[6:])

    #-------------------------------------------------------------------------
    def run(self, image):
        writer = {
          ANALYZE_FORMAT: self.writeAnalyze,
          NIFTI_SINGLE: self.writeNifti,
          NIFTI_DUAL: self.writeNifti,
        }.get(self.format, None)
        if writer is None: print "Unsupported output type: %s"%self.format
        writer(image)
