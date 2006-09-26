from recon.imageio import writeImage
from recon.util import castData
from recon.operations import Operation, Parameter

ANALYZE_FORMAT = "analyze"
NIFTI_DUAL = "nifti-dual"
NIFTI_SINGLE = "nifti-single"
MAGNITUDE_TYPE = "magnitude"
COMPLEX_TYPE = "complex"

##############################################################################
class WriteImage (Operation):
    "Write image to the filesystem."

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
        description="Output datatype (complex or magnitude). Default is "\
                    "magnitude."))

    #-------------------------------------------------------------------------
    def writeAnalyze(self, image):
        from recon import analyze
        # convert to format-specific datatype constant
        if self.datatype == COMPLEX_TYPE:
            # misunderstanding, default to
            # magnitude type handled in next case
            if not hasattr(image.data, "imag"):
                self.datatype = MAGNITUDE_TYPE
            data_code = analyze.COMPLEX        
        if self.datatype == MAGNITUDE_TYPE:
            if hasattr(image.data, "imag"):
                image.data = abs(image.data)
            data_code = analyze.FLOAT

        # if the image data isn't of the desired type (analyze.FLOAT, or
        # analyze.COMPLEX), then cast it there.
        if data_code != analyze.typecode2datatype[image.data.typecode()]:
            image[:] = image.data.astype(analyze.datatype2typecode[data_code])
                                           
        analyze.writeImage(image, self.filename,
                           datatype=data_code,
                           targetdim=self.filedim,
                           suffix=self.suffix,
                           scale=1.0)
    #-------------------------------------------------------------------------
    def writeNifti(self, image):
        from recon import nifti
         # convert to format-specific datatype constant
        if self.datatype == COMPLEX_TYPE:
            # misunderstanding, default to
            # magnitude type handled in next case
            if not hasattr(image.data, "imag"):
                self.datatype = MAGNITUDE_TYPE
            data_code = nifti.COMPLEX        
        if self.datatype == MAGNITUDE_TYPE:
            if hasattr(image.data, "imag"):
                image.data = abs(image.data)
            data_code = nifti.FLOAT

        # if the image data isn't of the desired type (nifti.FLOAT, or
        # nifti.COMPLEX), then cast it there.
        if data_code != nifti.typecode2datatype[image.data.typecode()]:
            image[:] = image.data.astype(nifti.datatype2typecode[data_code])
            
        nifti.writeImage(image, self.filename,
                         datatype=data_code,
                         targetdim=self.filedim,
                         filetype=self.format[6:],
                         suffix=self.suffix,
                         scale=1.0)
    #-------------------------------------------------------------------------
    def run(self, image):
        writer = {
          ANALYZE_FORMAT: self.writeAnalyze,
          NIFTI_SINGLE: self.writeNifti,
          NIFTI_DUAL: self.writeNifti,
        }.get(self.format, None)
        if writer is None: print "Unsupported output type: %s"%self.format
        writer(image)
