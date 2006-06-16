"This module can write NIFTI files."
from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64,\
  Complex32, fromstring, reshape, product, array, sin, cos, pi, asarray, sign
import struct
import exceptions

from odict import odict
from imaging.util import struct_unpack, struct_pack, NATIVE, LITTLE_ENDIAN, \
     BIG_ENDIAN, euler2quat, qmult
from imaging.imageio import BaseImage
from imaging.analyze import _concatenate

# datatype is a bit flag into the datatype identification byte of the NIFTI
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

#q/sform codes
NIFTI_XFORM_UNKNOWN = 0
NIFTI_XFORM_SCANNER_ANAT = 1
NIFTI_XFORM_ALIGNED_ANAT = 2
NIFTI_XFORM_TALAIRACH = 3
NIFTI_XFORM_MNI_152 = 4

#slice codes:
NIFTI_SLICE_UNKNOWN = 0
NIFTI_SLICE_SEQ_INC = 1
NIFTI_SLICE_SEQ_DEC = 2
NIFTI_SLICE_ALT_INC = 3
NIFTI_SLICE_ALT_DEC = 4

# map datatype to number of bits per pixel (or voxel)
datatype2bitpix = {
  BYTE: 8,
  SHORT: 16,
  INTEGER: 32,
  FLOAT: 32,
  DOUBLE: 64,
  COMPLEX: 64,
}

# map NIFTI datatype to Numeric typecode
datatype2typecode = {
  BYTE: Int8,
  SHORT: Int16,
  INTEGER: Int32,
  FLOAT: Float32,
  DOUBLE: Float64,
  COMPLEX: Complex32}

# map Numeric typecode to NIFTI datatype
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
    ('dim_info','B'),
    # these are short dim[8]
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
    # these 8 float are float pixdim[8] (pixdim[0] encodes qfac)
    ('qfac','f'),
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
    ('slice_code','B'),
    ('xyzt_units','B'),
    ('cal_max','f'),
    ('cal_min','f'),
    ('slice_duration','f'),
    ('toffset','f'),
    ('glmax','i'),
    ('glmin','i'),
    ('descrip','80s'),
    ('aux_file','24s'),
    ('qform_code','h'),
    ('sform_code','h'),
    ('quatern_b','f'),
    ('quatern_c','f'),
    ('quatern_d','f'),
    ('qoffset_x','f'),
    ('qoffset_y','f'),
    ('qoffset_z','f'),
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

##############################################################################
class NiftiImage (BaseImage):
    """
    Loads an image from a NIFTI file as a BaseImage
    """
    def __init__(self, filestem):
        self.load_header(filestem)
        self.load_image(filestem)

    def load_header(self, filestem):
        try:
            fp = open(filestem+".hdr", 'r')
        except exceptions.IOError:
            try:
                fp = open(filestem+".nii", 'r')
            except exceptions.IOError:
                raise "no NIFTI file found with this name: %s"%filestem
            self.filetype = 'single'
        else: self.filetype = 'dual'
#        print self.filetype
        byte_order = LITTLE_ENDIAN
        reported_length = struct_unpack(fp, byte_order, field_formats[0:1])[0]
        if reported_length != HEADER_SIZE: byte_order = BIG_ENDIAN
        fp.seek(0)
        # unpack all header values
        values = struct_unpack(fp, byte_order, field_formats)
        fp.close()
        # now load values into self
        map(self.__setattr__, struct_fields.keys(), values)
        # sanity check? why not
        if (self.filetype == 'single' and self.magic != 'n+1\x00') \
           or (self.filetype == 'dual' and self.magic != 'ni1\x00'):
            print "Got %s NIFTI file, but read %s magic string"%\
                  (self.filetype, self.magic)
            
    def load_image(self, filestem):
        if self.filetype == 'single':
            fp = open(filestem+".nii", 'r')
            fp.seek(self.vox_offset)
        else:
            fp = open(filestem+".img", 'r')

        numtype = datatype2typecode[self.datatype]
        dims = self.tdim and (self.tdim, self.zdim, self.ydim, self.xdim) \
                          or (self.zdim, self.ydim, self.xdim)
        datasize = (self.bitpix/8)*product(dims)
        image = fromstring(fp.read(datasize), numtype)
        self.setData(reshape(image, dims))
        fp.close()


##############################################################################
class NiftiWriter (object):
    """
    Writes an image in single- or dual-file NIFTI format.
    MANY useful features of this file format are not yet utilized!
    """

    #[STATIC]
    _defaults_for_fieldname = {'sizeof_hdr': HEADER_SIZE, 'scale_factor':1.}
    #[STATIC]
    _defaults_for_descriptor = {'i': 0, 'h': 0, 'f': 0., \
                                'c': '\0', 's': '', 'B': 0}

    def __init__(self, image, datatype=None, filetype="single", params=dict()):
        self.image = image
        self.params = params
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
            file(fname,'a').write(default_extension)
            file(fname,'a').write(self.image.data.tostring())
        else:
            headername, imagename = "%s.hdr"%filestem, "%s.img"%filestem
            file(headername,'w').write(self.make_hdr())
            file(imagename, 'w').write(self.image.data.tostring())
           

    #-------------------------------------------------------------------------
    def make_hdr(self):
        "Pack a NIFTI format header."
        # (un)rotation is handled like this: take the image as transformed with
        # 2 rotations, S, Rb (S is the xform from scanner space into the
        # data-ordering in the FID file, Rb is the xform applied by slicing
        # coronal-wise, sagital-wise, etc)
        # then I = Rb(psi)*Rb(theta)*Rb(phi)*S*I_real
        # The goal is to encode a quaternion which does this inverse:
        # R = R(-S)*R(-phi)*R(-theta)*R(-psi)
        #
        # the euler angles for inverting S have been determined to be
        # (phi=pi, theta=0, psi=-pi/2)
        # The euler angles for slicing rotations are from procpar
        # To avoid the chance of interpolation, these angles are "normalized"
        # to the closest multiple of pi/2 (ie 22 degrees->0, 49 degrees->90)

        image = self.image
        pdict = self.params
        phi,theta,psi = map(lambda x: (pi/2)*int((x+sign(x)*45.)/90),
                            (pdict.phi[0], pdict.theta[0], pdict.psi[0]))

        Qscanner = euler2quat(phi=pi, psi=-pi/2)
        Qobl = qmult(euler2quat(phi=-phi),qmult(euler2quat(theta=-theta),
                                                euler2quat(psi=-psi)))
        Qform = qmult(Qscanner,Qobl)
        imagevalues = {
          'dim_info': (3<<4 | 2<<2 | 1),
          'slice_code': NIFTI_SLICE_SEQ_INC,
          'datatype': self.datatype,
          'bitpix': datatype2bitpix[self.datatype],
          'ndim': image.ndim,
          'xdim': image.xdim,
          'ydim': image.ydim,
          # including t,z info should probably depend on targetdims
          'zdim': image.zdim,
          'tdim': image.tdim,
          'xsize': image.xsize,
          'ysize': image.ysize,
          'zsize': image.zsize,
          'tsize': image.tsize,
          'xyzt_units': (NIFTI_UNITS_MM | NIFTI_UNITS_SEC),
          'qfac': -1.0, # this makes all rotations in right-handed terms now!
          'qform_code': NIFTI_XFORM_SCANNER_ANAT,
          'quatern_b': Qform[1],
          'quatern_c': Qform[2],
          'quatern_d': Qform[3],
          'qoffset_x': float(image.xsize*image.xdim/2.),
          'qoffset_y': float(image.ysize*image.ydim/2.),
          'qoffset_z': 0.0,
          }
        
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
        NiftiWriter(subimage, datatype=datatype, filetype=filetype,
                    params=image._procpar).write(substem)

#-----------------------------------------------------------------------------
def writeImageDual(image, filestem, datatype=None, targetdim=None):
    writeImage(image, filestem, datatype, targetdim, filetype="dual")

#-----------------------------------------------------------------------------
def readImage(filestem): return NiftiImage(filestem)
