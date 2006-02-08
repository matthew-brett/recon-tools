"This module implements details of the Analyze7.5 file format."
from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64,\
  Complex32, fromstring, reshape
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

HEADER_SIZE = 348
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
field_formats = struct_fields.values()

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

        # Determine byte order of the header.  The first header element is the
        # header size.  It should always be 384.  If it is not then you know
        # you read it in the wrong byte order.
        byte_order = LITTLE_ENDIAN
        reported_length = struct_unpack(file(filename),
          byte_order, field_formats[0:1])[0]
        if reported_length != HEADER_SIZE: byte_order = BIG_ENDIAN

        # unpack all header values
        values = struct_unpack(file(filename), byte_order, field_formats)

        # now load values into self
        map(self.__setattr__, struct_fields.keys(), values)

    #-------------------------------------------------------------------------
    def load_image(self, filename):

        # bytes per pixel
        bytepix = self.bitpix/8
        numtype = datatype2typecode[self.datatype]
        #new_numtype = self.datatype==COMPLEX and Complex32 or Float32
        dims = self.tdim and (self.tdim, self.zdim, self.ydim, self.xdim)\
                          or (self.zdim, self.ydim, self.xdim)
        datasize = bytepix * reduce(lambda x,y: x*y, dims)
        image = fromstring(file(filename).read(datasize),numtype)#\
                #.astype(new_numtype)
        self.setData(reshape(image, dims))


##############################################################################
class AnalyzeWriter (object):
    """
    Write a given image into a single Analyze7.5 format hdr/img pair.
    """

    #[STATIC]
    _defaults_for_fieldname = {'sizeof_hdr': HEADER_SIZE, 'scale_factor':1.}
    #[STATIC]
    _defaults_for_descriptor = {'i': 0, 'h': 0, 'f': 0., 'c': '\0', 's': ''}

    #-------------------------------------------------------------------------
    def __init__(self, image, datatype=None):
        self.image = image
        self.datatype = datatype or typecode2datatype[image.data.typecode()]

    #-------------------------------------------------------------------------
    def _default_field_value(self, fieldname, fieldformat):
        "[STATIC] Get the default value for the given field."
        return self._defaults_for_fieldname.get(fieldname, None) or \
               self._defaults_for_descriptor[fieldformat[-1]]

    #-------------------------------------------------------------------------
    def write(self, filestem):
        "Write ANALYZE format header, image file pair."
        headername, imagename = "%s.hdr"%filestem, "%s.img"%filestem
        self.write_hdr(headername)
        self.write_img(imagename)

    #-------------------------------------------------------------------------
    def write_hdr(self, filename):
        "Write ANALYZE format header (.hdr) file."
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
          'xsize': image.xsize,
          'xsize': image.xsize,
          'xsize': image.xsize}

        def fieldvalue(fieldname, fieldformat):
            return imagevalues.get(fieldname) or\
                   self._default_field_value(fieldname, fieldformat)

        fieldvalues = [fieldvalue(*field) for field in struct_fields.items()]
        header = struct_pack(NATIVE, field_formats, fieldvalues)
        file(filename,'w').write(header)

    #-------------------------------------------------------------------------
    def write_img(self, filename):
        "Write ANALYZE format image (.img) file."
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

        # Write the image file.
        f = file( filename, "w" )
        f.write( imagedata.tostring() )
        f.close()

#-----------------------------------------------------------------------------
def _concatenate(listoflists):
    "Flatten a list of lists by one degree."
    finallist = []
    for sublist in listoflists: finallist += sublist
    return finallist

#-----------------------------------------------------------------------------
def writeImage(image, filestem, datatype=None, targetdim=3):
    """
    Write the given image to the filesystem as one or more Analyze7.5 format
    hdr/img pairs.
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

    for subimage, substem in images_and_names(image, filestem, targetdim):
        AnalyzeWriter(subimage, datatype=datatype).write(substem)

#-----------------------------------------------------------------------------
def readImage(filename): return AnalyzeImage(filename)
