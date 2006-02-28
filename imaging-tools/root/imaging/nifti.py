"This module can write NIFTI files."
from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64,\
  Complex32, fromstring, reshape
import struct
from analyze import _concatenate
from odict import odict
from imaging.imageio import BaseImage

# maximum numeric range for some smaller data types
maxranges = {
  Int8:  255.,
  Int16: 32767.,
  Int32: 2147483648.}

# datatype is a bit flag into the datatype identification byte of the Analyze
# header. 
BYTE = 2
SHORT = 4
INTEGER = 8
FLOAT = 16
COMPLEX = 32
DOUBLE = 64

# some bit-mask codes
NIFTI_UNITS_UNKNOWN = 0
NIFTI_UNITS_METER = 1
NIFTI_UNITS_MM = 2
NIFTI_UNITS_MICRON = 3
NIFTI_UNITS_SEC = 8
NIFTI_UNITS_MSEC = 16
NIFTI_UNITS_USEC = 24

# map datatype to number of bits per pixel (or voxel)
datatype2bitpix = {
  BYTE: 8,
  SHORT: 16,
  INTEGER: 32,
  FLOAT: 32,
  DOUBLE: 64,
  COMPLEX: 64,
}

# map Analyze datatype to Numeric typecode
datatype2typecode = {
  BYTE: Int8,
  SHORT: Int16,
  INTEGER: Int32,
  FLOAT: Float32,
  DOUBLE: Float64,
  COMPLEX: Complex32}

# map Numeric typecode to Analyze datatype
typecode2datatype = \
  dict([(v,k) for k,v in datatype2typecode.items()])

HEADER_SIZE = 348
struct_fields = odict((
    ('sizeof_hdr','i'),
    ('data_type','10s'),
    ('db_name','18s'),
    ('extents','i'),
    ('session_error','h'),
    ('regular','c'),
    ('dim_info','c'),
    ('ndim','h'),
    ('xdim','h'),
    ('ydim','h'),
    ('zdim','h'),
    ('tdim','h'),
    ('dim5','h'),
    ('dim6','h'),
    ('dim7','h'),
    ('intent_p1','f'),
    ('intent_p2','f'),
    ('intent_p3','f'),
    ('intent_code','h'),
    ('datatype', 'h'),
    ('bitpix','h'),
    ('slice_start', 'h'),
    ('pixdim0','f'),
    ('xsize','f'),
    ('ysize','f'),
    ('zsize','f'),
    ('tsize','f'),
    ('pixdim5','f'),
    ('pixdim6','f'),
    ('pixdim7','f'),
    ('vox_offset','f'),
    ('scl_slope','f'),
    ('scl_inter','f'),
    ('slice_end','h'),
    ('slice_code','s'),
    ('xyzt_units','s'),
    ('cal_max','f'),
    ('cal_min','f'),
    ('slice_duration','f'),
    ('toffset','f'),
    ('glmax','i'),
    ('glmin','i'),
    ('descrip','80s'),
    ('aux_file','24s'),
    ('gform_code','h'),
    ('sform_code','h'),
    ('quatern_b','f'),
    ('quatern_c','f'),
    ('quatern_d','f'),
    ('quatern_x','f'),
    ('quatern_y','f'),
    ('quatern_z','f'),
    ('srow_x0','f'),
    ('srow_x1','f'),
    ('srow_x2','f'),
    ('srow_x3','f'),
    ('srow_y0','f'),
    ('srow_y1','f'),
    ('srow_y2','f'),
    ('srow_y3','f'),
    ('srow_z0','f'),
    ('srow_z1','f'),
    ('srow_z2','f'),
    ('srow_z3','f'),
    ('intent_name','16s'),
    ('magic','4s'),
))
field_formats = struct_fields.values()

#define a 4 blank bytes for a null extension
default_extension = struct.pack('l', 0)

# struct byte order constants
NATIVE = "="
LITTLE_ENDIAN = "<"
BIG_ENDIAN = ">"

def struct_format(byte_order, elements):
    return byte_order+" ".join(elements)
    
def struct_unpack(infile, byte_order, elements):
    format = struct_format(byte_order, elements)
    return struct.unpack(format, infile.read(struct.calcsize(format)))

def struct_pack(byte_order, elements, values):
    format = struct_format(byte_order, elements)
    return struct.pack(format, *values)


##############################################################################
class NiftiWriter (object):
    """
    Writes an image in single- or dual-file NIFTI format
    """

    #[STATIC]
    _defaults_for_fieldname = {'sizeof_hdr': HEADER_SIZE, 'scale_factor':1.}
    #[STATIC]
    _defaults_for_descriptor = {'i': 0, 'h': 0, 'f': 0., 'c': '\0', 's': ''}

    def __init__(self, image, datatype=None, filetype="single"):
        self.image = image
        self.filetype = filetype
        self.datatype = datatype or typecode2datatype[image.data.typecode()]
        self.sizeof_extension = len(default_extension)

    #-------------------------------------------------------------------------
    def _default_field_value(self, fieldname, fieldformat):
        "[STATIC] Get the default value for the given field."
        return self._defaults_for_fieldname.get(fieldname, None) or \
               self._defaults_for_descriptor[fieldformat[-1]]

    #-------------------------------------------------------------------------
    def write(self, filestem):
        "Write NIFTI format header, image in single file or paired files."
        if self.filetype=="single":
            fname = "%s.nii"%filestem
            file(fname,'w').write(self.make_hdr())
            #write extension? 
            #file(fname,'a').write(default_extension)
            file(fname,'a').write(self.format_img())
        else:
            headername, imagename = "%s.hdr"%filestem, "%s.img"%filestem
            file(headername,'w').write(self.make_hdr())
            file(imagename, 'w').write(self.format_img())
           

    #-------------------------------------------------------------------------
    def make_hdr(self):
        "Pack a NIFTI format header."
        image = self.image
        imagevalues = {
          'datatype': self.datatype,
          'bitpix': datatype2bitpix[self.datatype],
          'ndim': image.ndim,
          'xdim': image.xdim,
          'ydim': image.ydim,
          'zdim': image.zdim,
          'tdim': image.tdim,
          'xsize': image.xsize,
          'ysize': image.ysize,
          'zsize': image.zsize,
          'tsize': image.tsize
          'xyzt_units': chr(NIFTI_UNITS_MM | NIFTI_UNITS_SEC)

        if self.filetype=='single':
            imagevalues['vox_offset'] = HEADER_SIZE + self.sizeof_extension
            imagevalues['magic'] = 'n+1'
        else:
            imagevalues['magic'] = 'ni1'

        def fieldvalue(fieldname, fieldformat):
            return imagevalues.get(fieldname) or\
                   self._default_field_value(fieldname, fieldformat)

        fieldvalues = [fieldvalue(*field) for field in struct_fields.items()]
        return struct_pack(NATIVE, field_formats, fieldvalues)

    #-------------------------------------------------------------------------
    def format_img(self):
        "Format image data to write to a NIFTI file."
        imagedata = self.image.data

        if self.datatype != COMPLEX: imagedata = abs(imagedata)

        # if requested datatype does not correspond to image datatype, cast
        if self.datatype != typecode2datatype[imagedata.typecode()]:
            typecode = datatype2typecode[self.datatype]

            # Make sure image values are within the range of the desired
            # data type
            if self.datatype in (BYTE, SHORT, INTEGER):
                maxval = amax(abs(imagedata).flat)
                if maxval == 0.: maxval = 1.e20
                maxrange = maxranges[typecode]

                # if out of desired bounds, perform scaling
                if maxval > maxrange: imagedata *= (maxrange/maxval)

            # cast image values to the desired datatype
            imagedata = imagedata.astype( typecode )
            print "CASTING"

        return imagedata.tostring()


#-----------------------------------------------------------------------------
def writeImage(image, filestem, datatype=None, targetdim=None, filetype="single"):
    """
    Write the given image to the filesystem as one or more NIFTI 1.1 format
    hdr/img pairs or single file format.
    @param filestem:  will be prepended to each hdr and img file.
    @param targetdim:  indicates the dimensionality of data to be written into
      a single hdr/img pair.  For example, if a volumetric time-series is
      given, and targetdim==3, then each volume will get its own file pair.
      Likewise, if targetdim==2, then every slice of each volume will get its
      own pair.
    """
    dimnames = {3:"volume", 2:"slice"}
    def images_and_names(image, stem, targetdim):
        # base case
        if targetdim >= image.ndim: return [(image, stem)]
        
        # recursive case
        subimages = tuple(image.subImages())
        substems = ["%s_%s%d"%(stem, dimnames[image.ndim-1], i)\
                    for i in range(len(subimages))]
        return _concatenate(
          [images_and_names(subimage, substem, targetdim)\
           for subimage,substem in zip(subimages, substems)])

    if targetdim is None: targetdim = image.ndim
    for subimage, substem in images_and_names(image, filestem, targetdim):
        NiftiWriter(subimage, datatype=datatype, filetype=filetype).write(substem)
