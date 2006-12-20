"This module can write NIFTI files."
import Numeric as N
import struct
import exceptions
import sys

from odict import odict
from recon.util import struct_unpack, struct_pack, NATIVE, euler2quat, qmult, \
     Quaternion, range_exceeds, volume_min, volume_max
from recon.imageio import ReconImage
from recon.analyze import datatype2bitpix, datatype2typecode, \
     typecode2datatype, byteorders, _construct_dataview

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
class NiftiImage (ReconImage):
    """
    Loads an image from a NIFTI file as a ReconImage
    """
    def __init__(self, filestem, target_dtype=None, vrange=()):
        self.load_header(filestem)
        self.load_image(filestem, target_dtype, vrange)
        
    #-------------------------------------------------------------------------
    def load_header(self, filestem):
        try:
            fp = open(filestem+".nii", 'r')
        except exceptions.IOError:
            try:
                fp = open(filestem+".hdr", 'r')
            except exceptions.IOError:
                raise IOError("no NIFTI file found with this name: %s"%filestem)
            self.filetype = 'dual'
        else: self.filetype = 'single'

        byte_order = byteorders[sys.byteorder]
        self.swapped = False
        reported_length = struct_unpack(fp, byte_order, field_formats[0:1])[0]
        if reported_length != HEADER_SIZE:
            byte_order = byteorders['swapped']
            self.swapped = True
        fp.seek(0)
        # unpack all header values
        values = struct_unpack(fp, byte_order, field_formats)
        fp.close()
        # now load values into self
        hd_vals = dict()
        map(hd_vals.__setitem__, struct_fields.keys(), values)
        # sanity check? why not
        if (self.filetype == 'single' and hd_vals['magic'] != 'n+1\x00') \
           or (self.filetype == 'dual' and hd_vals['magic'] != 'ni1\x00'):
            raise ValueError("Got file %s, but magic string is incorrect: %s"%\
                  (filestem, hd_vals['magic']))

        # These values are required to be a ReconImage
        (self.xdim, self.ydim, self.zdim, self.tdim) = \
                    (hd_vals['xdim'], hd_vals['ydim'],
                     hd_vals['zdim'], hd_vals['tdim'])
        (self.xsize, self.ysize, self.zsize, self.tsize) = \
                     (hd_vals['xsize'], hd_vals['ysize'],
                      hd_vals['zsize'], hd_vals['tsize'])
        (self.x0, self.y0, self.z0) = \
                  (hd_vals['qoffset_x'], hd_vals['qoffset_y'],
                   hd_vals['qoffset_z'])
        # what about orientation name?
        qb, qc, qd, qfac = \
            (hd_vals['quatern_b'], hd_vals['quatern_c'],
             hd_vals['quatern_d'], hd_vals['qfac'])
        self.orientation_xform = Quaternion(i=qb, j=qc, k=qd, qfac=qfac)
        self.vox_offset, self.datatype, self.bitpix = \
                         (hd_vals['vox_offset'], hd_vals['datatype'],
                          hd_vals['bitpix'])
        self.scaling,self.yinter = (hd_vals['scl_slope'],hd_vals['scl_inter'])
        if not self.scaling:
            self.scaling = 1.0
        
    #-------------------------------------------------------------------------
    def load_image(self, filestem, target_dtype, vrange):
        bytepix = self.bitpix/8
        numtype = datatype2typecode[self.datatype]
        byteoffset = 0
        if target_dtype and not range_exceeds(target_dtype, numtype):
            raise ValueError("the dynamic range of the desired datatype does "\
                             "not exceed that of the raw data")
        if self.filetype == 'single':
            fp = open(filestem+".nii", 'r')
            fp.seek(self.vox_offset)
        else:
            fp = open(filestem+".img", 'r')
        # need to cook tdim if vrange is set
        if self.tdim and vrange:
            vend = (vrange[1]<0 or vrange[1]>=self.tdim) \
                   and self.tdim-1 or vrange[1]
            vstart = (vrange[0] > vend) and vend or vrange[0]
            self.tdim = vend-vstart+1
            byteoffset = vstart*bytepix*N.product((self.zdim,
                                                   self.ydim,self.xdim))

        dims = self.tdim and (self.tdim, self.zdim, self.ydim, self.xdim) \
                          or (self.zdim, self.ydim, self.xdim)
        datasize = bytepix * N.product(dims)
        fp.seek(byteoffset, 1)
        image = N.fromstring(fp.read(datasize), numtype)
        if self.swapped: image = image.byteswapped()
        if target_dtype and target_dtype != numtype:
            image = (image*self.scaling+self.yinter).astype(target_dtype)
        self.setData(N.reshape(image, dims))
        fp.close()


##############################################################################
class NiftiWriter (object):
    """
    Writes an image in single- or dual-file NIFTI format.
    MANY useful features of this file format are not yet utilized!
    """

    #[STATIC]
    _defaults_for_fieldname = {'sizeof_hdr': HEADER_SIZE,
                               'scl_slope':1.}
    #[STATIC]
    _defaults_for_descriptor = {'i': 0, 'h': 0, 'f': 0., \
                                'c': '\0', 's': '', 'B': 0}

    def __init__(self, image, datatype=None, filetype="single", scale=1.0):
        self.image = image
        self.scaling = scale
        self.filetype = filetype
        self.datatype = datatype or typecode2datatype[image[:].typecode()]
        typecode = datatype2typecode[self.datatype]
        self._dataview = _construct_dataview(image[:].typecode(), self.datatype,
                                             self.scaling)
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
            file(fname,'a').write(self._dataview(self.image[:]).tostring())
        else:
            headername, imagename = "%s.hdr"%filestem, "%s.img"%filestem
            file(headername,'w').write(self.make_hdr())
            file(imagename, 'w').write(self._dataview(self.image[:]).tostring())
           

    #-------------------------------------------------------------------------
    def make_hdr(self):
        "Pack a NIFTI format header."
        # The NIFTI Q-form field is helpful for keeping track of anatomical
        # dimensions in the presence of rotations.
        #
        # (un)rotation is handled like this: take the image as transformed with
        # 2 rotations, Qs, Rb (Qs represents a transform between scanner space
        # and a right-handed interpretation of the data, Rb is the xform
        # applied by slicing coronal-wise, sagital-wise, etc with the slice
        # gradient).
        #     Then I_m = Qs*Rb(psi,theta,phi)*I_real
        # The goal is to encode a quaternion which corrects this rotation:
        #     Rc = inv(Rb)*inv(Qs)
        #     so I_real = Rc*I_m
        #
        # The euler angles for slicing rotations are from procpar
        # To avoid the chance of interpolation, these angles are "normalized"
        # to the closest multiple of pi/2 (ie 22 degrees->0, 49 degrees->90)
        # 

        image = self.image
        Qform = image.orientation_xform.Q
        Qform_mat = image.orientation_xform.tomatrix()
        qoffset = -N.dot(Qform_mat, N.asarray([image.x0, image.y0, image.z0]))

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
          'scl_slope': self.scaling,
          'xyzt_units': (NIFTI_UNITS_MM | NIFTI_UNITS_SEC),
          'qfac': image.orientation_xform.qfac,
          'qform_code': NIFTI_XFORM_SCANNER_ANAT,
          'quatern_b': Qform[0],
          'quatern_c': Qform[1],
          'quatern_d': Qform[2],
          'qoffset_x': qoffset[0],
          'qoffset_y': qoffset[1],
          'qoffset_z': qoffset[2],
          'cal_min': float(volume_min(abs(image[:]))),
          'cal_max': float(volume_max(abs(image[:]))),
          }
        
        if self.filetype=='single':
            imagevalues['vox_offset'] = HEADER_SIZE + self.sizeof_extension
            imagevalues['magic'] = 'n+1'
        else:
            imagevalues['magic'] = 'ni1'

        def fieldvalue(fieldname, fieldformat):
            return imagevalues.get(fieldname) or \
                   self._default_field_value(fieldname, fieldformat)

        fieldvalues = [fieldvalue(*field) for field in struct_fields.items()]
        return struct_pack(NATIVE, field_formats, fieldvalues)

#-----------------------------------------------------------------------------
def readImage(filestem, **kwargs): return NiftiImage(filestem, **kwargs)
