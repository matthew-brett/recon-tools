"This module implements details of the Analyze7.5 file format."
from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64, Complex32
import struct

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

HEADER_SIZE = 384
struct_fields = odict((
    ('sizeof_hdr','i'),
    ('data_type','10s'),
    ('db_name','18s'),
    ('extents','i'),
    ('session_error','h'),
    ('regular','c'),
    ('hkey_un0','c'),
    ('ndim','h'),
    ('xdim','h'),
    ('ydim','h'),
    ('zdim','h'),
    ('tdim','h'),
    ('dim5','h'),
    ('dim6','h'),
    ('dim7','h'),
    ('vox_units','4s'),
    ('cal_units','8s'),
    ('unused1','h'),
    ('datatype','h'),
    ('bitpix','h'),
    ('dim_un0','h'),
    ('pixdim0','f'),
    ('xsize','f'),
    ('ysize','f'),
    ('zsize','f'),
    ('tsize','f'),
    ('pixdim5','f'),
    ('pixdim6','f'),
    ('pixdim7','f'),
    ('vox_offset','f'),
    ('scale_factor','f'),
    ('funused2','f'),
    ('funused3','f'),
    ('cal_max','f'),
    ('cal_min','f'),
    ('compressed','i'),
    ('verified','i'),
    ('glmax','i'),
    ('glmin','i'),
    ('descrip','80s'),
    ('aux_file','24s'),
    ('orient','c'),
    ('x0','h'),
    ('y0','h'),
    ('z0','h'),
    ('sunused','4s'),
    ('generated','10s'),
    ('scannum','10s'),
    ('patient_id','10s'),
    ('exp_date','10s'),
    ('exp_time','10s'),
    ('hist_un0','3s'),
    ('views','i'),
    ('vols_added','i'),
    ('start_field','i'),
    ('field_skip','i'),
    ('omax','i'),
    ('omin','i'),
    ('smax','i'),
    ('smin','i'),
))

# struct byte order constants
NATIVE = "="
LITTLE_ENDIAN = "<"
BIG_ENDIAN = ">"

def struct_format(byte_order, elements):
    return byte_order+" ".join(elements)
    
def struct_unpack(infile, byte_order, elements):
    format = struct_format(byte_order, elements)
    return struct.unpack(format, infile.read(struct.calcsize(format)))


##############################################################################
class AnalyzeImage (BaseImage):
    """
    Image interface conformant Analyze7.5 file reader.
    """

    #-------------------------------------------------------------------------
    def __init__(self, filestem):
        self.load_header(filestem+".hdr")
        self.load_image(filestem+".img")

    #-------------------------------------------------------------------------
    def load_header(self, filename):
        "Load Analyze7.5 header from the given filename"
        field_formats = struct_fields.values()

        # Determine byte order of the header.  The first header element is the
        # header size.  It should always be 384.  If it is not then you know
        # you read it in the wrong byte order.
        byte_order = LITTLE_ENDIAN
        reported_length = struct_unpack(file(filename),
          byte_order, field_formats[0:1])[0]
        if reported_length != HEADER_SIZE: byte_order = BIG_ENDIAN

        # unpack all header values
        values = struct_unpack(file(filename), byte_order, field_formats)
        print len(field_formats),len(values)

        # now load values into self
        map(self.__setattr__, zip(field_formats, values))

    #-------------------------------------------------------------------------
    def load_image(self, filename):

        # bytes per pixel
        bytepix = self.bitpix/8
        numtype = datatype2typecode[(self.datatype,self.bitpix)]
        new_numtype = self.datatype==COMPLEX and Complex32 or Float32
        datasize = xdim*ydim*zdim*tdim*bytepix
        image = fromstring(file(filename).read(datasize),numtype)\
                .astype(new_numtype)
        dims = self.tdim and (self.tdim, self.zdim, self.ydim, self.xdim)\
                          or (self.zdim, self.ydim, self.xdim)
        self.setData(reshape(image, dims))


##############################################################################
class AnalyzeWriter (object):
    """
    Write images in Analyze7.5 format.
    """

    _defaults_for_fieldname = {'sizeof_hdr': HEADER_SIZE, 'scale_factor':1.}
    _defaults_for_descriptor = {'i': 0, 'h': 0, 'f': 0., 'c': '', 's': ''}

    #-------------------------------------------------------------------------
    def __init__(self, image, datatype=None):
        self.image = image
        self.datatype = datatype or typecode2datatype[image.data.typecode()]

    #-------------------------------------------------------------------------
    def _default_field_value(self, fieldname, descriptor):
        return self._defaults_for_fieldname.get(fieldname, None) or \
               self._defaults_for_descriptor[descriptor[-1]]

    #-------------------------------------------------------------------------
    def write(self, filestem):
        "Write ANALYZE format header, image file pair."
        headername, imagename = "%s.hdr"%filestem, "%s.img"%filestem
        self.write_header(headername)
        self.write_image(imagename)

    #-------------------------------------------------------------------------
    def write_header(self, filename):
        "Write ANALYZE format header (.hdr) file."
        image = self.image
        values = {
          'datatype': self.datatype,
          'bitpix': datatype2bitpix[self.datatype],
          'ndim': image.ndim,
          'xdim': image.xdim,
          'ydim': image.ydim,
          'zdim': image.zdim,
          'tdim': image.tdim,
          'xsize': image.xsize,
          'xsize': image.xsize,
          'xsize': image.xsize,
          'xsize': image.xsize}
        datatype = self.datatype
        scale_factor = 1.
        bitpix = datatype2bitpix[datatype]
        cd = sd = " "
        hd = id = 0
        fd = 0.

        format = "=i 10s 18s i h 1s 1s 8h 4s 8s h h h h 8f f f f f f f i i 2i 80s 24s 1s 3h 4s 10s 10s 10s 10s 10s 3s i i i i 2i 2i"
        # why is TR used...?
        binary_header = struct.pack(
          format, 348, sd, sd, id, hd, cd, cd, image.ndim, image.xdim,
          image.ydim, image.zdim, image.tdim,
          hd, hd, hd, sd, sd, hd, datatype, bitpix, hd, fd, image.xsize,
          image.ysize, image.zsize, image.tsize, fd, fd, fd, fd, scale_factor,
          fd, fd, fd, fd, id, id, id, id, sd, sd, cd, image.x0, image.y0,
          image.z0, sd, sd, sd, sd, sd, sd, sd, id, id, id, id, id, id, id, id)
        f = open(filename,'w')
        f.write(binary_header)
        f.close()

    #-------------------------------------------------------------------------
    def write_image(self, filename):
        "Write ANALYZE format image (.img) file."
        imagedata = self.image.data

        # if requested datatype does not correspond to image datatype, cast
        if self.datatype != typecode2datatype[imagedata.typecode()]:
            typecode = datatype2typecode[self.datatype]

            # Make sure image values are within the range of the desired datatype
            if self.datatype in (BYTE, SHORT, INTEGER):
                maxval = amax(abs(imagedata).flat)
                if maxval == 0.: maxval = 1.e20
                maxrange = maxranges[typecode]

                # if out of desired bounds, perform scaling
                if maxval > maxrange: imagedata *= (maxrange/maxval)

            # cast image values to the desired datatype
            imagedata = imagedata.astype( typecode )

        if self.datatype != COMPLEX: imagedata = abs(imagedata)

        # Write the image file.
        f = file( filename, "w" )
        f.write( imagedata.tostring() )
        f.close()


#-----------------------------------------------------------------------------
def writeImage(image, filename, datatype=None):
    writer = AnalyzeWriter(image, datatype=datatype)
    writer.write(filename)

#-----------------------------------------------------------------------------
def readImage(filename): pass
