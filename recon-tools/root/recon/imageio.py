import numpy as N

from odict import odict
from recon.util import import_from, Quaternion, integer_ranges, scale_data

# module-private dict specifying available image readers
_readers = odict((
    ("analyze", ("recon.analyze","readImage")),
    ("nifti", ("recon.nifti","readImage")),
    ("fid", ("recon.scanners.varian","FidImage")),
    ("fdf", ("recon.scanners.varian","FDFImage"))))
available_readers = _readers.keys()

# module-private dict specifying available image writers
_writers = odict((
    ("analyze", ("recon.analyze","writeImage")),
    ("nifti-single", ("recon.nifti","writeImage")),
    ("nifti-dual", ("recon.nifti","writeDualImage")),
    ))

available_writers = _writers.keys()

recon_output2dtype = odict((
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
output_datatypes = recon_output2dtype.keys()

#-----------------------------------------------------------------------------
def get_dims(data):
    """
    Extract ndim, tdim, kdim, jdim, and idim from data shape.
    @return: (ndim, tdim, kdim, jdim, idim)
    """
    # In case the data dimension size for t or z are 1,
    # let's not reflect this in the image properties. 
    shape = data.shape
    while shape[0] < 2:
        shape = shape[1:]
    ndim = len(shape)
    if ndim < 2 or ndim > 4:
        raise ValueError("data shape %s must be 2, 3, or 4 dimensional"%shape)
    return (ndim,) + (0,)*(4-ndim) + shape

##############################################################################
class DataChunk (object):
    """
    A sub-iterator with N-dimensional data; can offer up further
    DataChunks with (N-1)-dimensional data
    """
    def __init__(self, data, num):
        self._data = data
        self.num = num
        self.shape = data.shape

    def __getitem__(self, slicer):
        return self._data[slicer]

    def __setitem__(self, slicer, data):
        self._data[slicer] = N.asarray(data).astype(self._data.dtype)
    
    def __iter__(self):
        if len(self.shape) > 1:
            iternum = 0
            while iternum < self.shape[0]:
                yield DataChunk(self._data[iternum], iternum)
                iternum += 1
            raise StopIteration
        else:
            raise ValueError("can't iterate through a 1D array")

##############################################################################
class ReconImage (object):
    """
    Interface definition for any image in Recon Tools.
    This class of images will be able to go through many of the available ops.

    This class of images can be exported to some medical imaging formats.
    
    Attributes:
      _data:  2, 3, or 4 dimensional matrix representing a slice, single
             volume, or a timecourse of volumes.
      ndim:  number of dimensions
      tdim:  number of volumes in a timecourse
      kdim:  number of slices per volume
      jdim:  number of rows per slice
      idim:  number of columns per row
      isize: spacial width of array columns
      jsize: spacial height of array rows
      ksize: spacial slice thickness (3rd dim of array)
      tsize: duration of each time-series volume (4th dim of array)
      x0: x coordinate of xyz offset
      y0: y coordinate of xyz offset
      z0: z coordinate of xyz offset
      orientation: name of the orientaion (coronal, axial, etc)
      orientation_xform: quaternion describing the orientation

    capabilities provided:
      volume/slice slicing
      fe/pe slicing
      __getitem__, __setitem__
      data xform (abs, real, imag, etc)
    """

    #-------------------------------------------------------------------------
    def __init__(self, data, isize, jsize, ksize, tsize, offset=None,
                 scaling=None, orient_xform=None, orient_name=None):
        """
        Construct a ReconImage with at least data, isize, jsize, ksize,
        and tsize known. Optional information are an offset 3-tuple
        specifying (x0,y0,z0), a Quaternion object representing the
        transformation of this data to neurological orientation
        (+X,+Y,+Z) = (Right,Anterior,Superior), and a name for the data's
        orientation (used for ANALYZE format output).
        """
        self.setData(data)
        self.isize, self.jsize, self.ksize, self.tsize = \
          (isize, jsize, ksize, tsize)

        self.orientation_xform = orient_xform or Quaternion()
        self.orientation = orient_name or ""

        # offset should be the (x,y,z) offset in xyz-space
        xform = self.orientation_xform.tomatrix()
        (self.x0, self.y0, self.z0) = \
                  offset or \
                  -N.dot(xform, N.array([self.isize*self.idim/2.,
                                         self.jsize*self.jdim/2.,
                                         self.ksize*self.kdim/2.]))
        

        self.scaling = scaling or 1.0

    #-------------------------------------------------------------------------
    def info(self):
        print "ndim =",self.ndim
        print "idim =",self.idim
        print "jdim =",self.jdim
        print "kdim =",self.kdim
        print "tdim =",self.tdim
        print "isize =",self.isize
        print "jsize =",self.jsize
        print "ksize =",self.ksize
        print "x0 =",self.x0
        print "y0 =",self.y0
        print "z0 =",self.z0
        print "data.shape =",self.data.shape
        print "data.dtype =",self.data.dtype

    #-------------------------------------------------------------------------
    def setData(self, data):
        "Inform self about dimension info from the data array"
        self.data = data
        self.ndim, self.tdim, self.kdim, self.jdim, self.idim = get_dims(data)
        self.shape = (self.tdim, self.kdim, self.jdim, self.idim)
        while self.shape[0] < 2:
            self.shape = self.shape[1:]
        self.data.shape = self.shape

    #-------------------------------------------------------------------------
    def concatenate(self, image, axis=0, newdim=False):
        """Stitch together two images along a given axis, possibly
        creating a new dimension
        """
        self_sizes = (self.isize, self.jsize, self.ksize)
        image_sizes = (image.isize, image.jsize, image.ksize)

        # pixel sizes must match
        if self_sizes != image_sizes:
            raise ValueError(
              "won't concatenate images with different pixel sizes: %s != %s"%\
              (self_sizes, image_sizes))

        if newdim:
            newdata = N.asarray((self[:], image.data))
        else:
            if len(self.shape) > len(image.shape):
                newdata = N.concatenate((self[:], image[(None,)]))
            else:
                newdata = N.concatenate((self[:], image.data), axis)
        return self._subimage(newdata)

    #-------------------------------------------------------------------------
    def __iter__(self):
        "Handles iteration over the image--always yields a 3D DataChunk"
        # want to iterate over volumes, if tdim=0, then nvol = 1
        if len(self.shape) > 3:
            for t in range(self.tdim):
                yield DataChunk(self[t], t)
            raise StopIteration
        else:
            yield DataChunk(self[:], 0)
            raise StopIteration
    #-------------------------------------------------------------------------
    def __getitem__(self, slicer):
        if type(slicer) is type(()) and len(slicer) > self.ndim:
            nfakes = len(slicer)-self.ndim
            slicer = (None,)*(nfakes) + slicer[nfakes:]
        return self.data[slicer]
    #-------------------------------------------------------------------------
    def __setitem__(self, slicer, newdata):
        ndata = N.asarray(newdata)
        if ndata.dtype.char.isupper() and self.data.dtype.char.islower():
            print "warning: losing information on complex->real cast!"
        if type(slicer) is type(()) and len(slicer) > self.ndim:
            nfakes = len(slicer)-self.ndim
            slicer = (None,)*(nfakes) + slicer[nfakes:]
        self.data[slicer] = ndata.astype(self.data.dtype)
    #-------------------------------------------------------------------------
    def __mul__(self, a):
        self[:] = self[:]*a
    #-------------------------------------------------------------------------
    def __div__(self, a):
        self[:] = self[:]/a
    #-------------------------------------------------------------------------
    def _subimage(self, data):        
        return ReconImage(data,
                          self.isize, self.jsize, self.ksize, self.tsize,
                          offset=(self.x0, self.y0, self.z0),
                          scaling=self.scaling,
                          orient_xform=self.orientation_xform,
                          orient_name=self.orientation)

    #-------------------------------------------------------------------------
    def subImage(self, subnum):
        "returns subnum-th sub-image with dimension ndim-1"
        return self._subimage(self.data[subnum])

    #-------------------------------------------------------------------------
    def subImages(self):
        "yeilds all images of dimension self.ndim-1"
        if len(self.shape) < 2:
            raise StopIteration("can't iterate subdimensions of a 2D image")
        for subnum in xrange(self.shape[0]):
            yield self.subImage(subnum)

    #-------------------------------------------------------------------------
    def resize(self, newsize_tuple):
        """
        resize/reshape the data, non-destructively if the number of
        elements doesn't change
        """
        if N.product(newsize_tuple) == N.product(self.shape):
            self.data = N.reshape(self.data, tuple(newsize_tuple))
        else:
            self.data.resize(tuple(newsize_tuple))
        self.setData(self[:])

    #-------------------------------------------------------------------------
    def writeImage(self, filestem, format_type="analyze",
                   datatype="magnitude", **kwargs):
        """
        Export the image object in a medical file format (ANALYZE or NIFTI).
        format_type is one of the internal file format specifiers, which
        are currently %s.
        possible keywords are:
        datatype -- a datatype identifier, supported by the given format
        targetdim -- number of dimensions per file
        filetype -- differentiates single + dual formats for NIFTI
        suffix -- over-ride default suffix style (eg volume0001)

        If necessary, a scaling is found for integer types
        """%(" ; ".join(available_writers))

        new_dtype = recon_output2dtype.get(datatype.lower(), None)
        if new_dtype is None:
            raise ValueError("Unsupported data type: %s"%datatype)

        if new_dtype in integer_ranges.keys():
            scale = float(scale_data(self[:], new_dtype))
        else:
            scale = float(1.0)

        _write(self, filestem, format_type, scale=scale,
               dtype=new_dtype,**kwargs)

#-----------------------------------------------------------------------------
def get_reader(format):
    "Return an image file reader for the specified format."
    readerspec = _readers.get(format)
    if readerspec is None:
        raise ValueError("Reader '%s' not found.  Avaliable readers are: %s"%\
          (format, ", ".join(available_readers)))
    return import_from(*readerspec)
#-----------------------------------------------------------------------------
def readImage(filename, format, datatype=None, **kwargs):
    "Load an image in the specified format from the given filename."
    if datatype and datatype not in recon_output2dtype.keys():
        raise ValueError("Unsupported data type: %s"%datatype)
    kwargs['target_dtype'] = recon_output2dtype.get(datatype, None)
    return get_reader(format)(clean_name(filename), **kwargs)

#-----------------------------------------------------------------------------
def get_writer(format_type):
    "Return an image file writer method for the Recon output type"
    writerspec = _writers.get(format_type)
    if writerspec is None:
        raise ValueError("Writer method for '%s' not found. "\
                         "Avaliable writers are: %s"%\
                         (format_type, ", ".join(available_writers)))
    return import_from(*writerspec)
#-----------------------------------------------------------------------------
def _write(image, filestem, format_type, dtype=None, targetdim=None,
           suffix=None, scale=1.0):
    """
    Given a RTools format_type description (eg, nifti-single), a numpy
    dtype in dtype, and a filename, write an appropriate file.
    Other keyword args:
      targetdim -- number of dimensions per output file (2, 3, or 4)
      suffix -- over-rides normal _volume000n suffix (for targetdim=3 only)
      scale -- scaling factor for integer output dtypes
    """

    Writer = get_writer(format_type)
    dimnames = {3: "volume", 2: "slice"}
    def images_and_names(image, stem, targetdim, suffix=None):
        # base case
        if targetdim >= image.ndim: return [(image, stem)]
        
        # recursive case
        subimages = tuple(image.subImages())
        if suffix is not None:
            substems = ["%s"%(stem,)+suffix%(i,) \
                        for i in range(len(subimages))]
        else:
            substems = ["%s_%s%04d"%(stem, dimnames[image.ndim-1], i)\
                        for i in range(len(subimages))]
        return _concatenate(
          [images_and_names(subimage, substem, targetdim)\
           for subimage,substem in zip(subimages, substems)])
    if suffix is not None: targetdim = 3
    if targetdim is None: targetdim = image.ndim
    for subimage, substem in images_and_names(image,filestem,targetdim,suffix):
        Writer(subimage, substem, dtype=dtype, scale=scale)

#-----------------------------------------------------------------------------
def clean_name(fname):
    pruned_exts = ['nii', 'hdr', 'img']
    if fname.rsplit('.')[-1] in pruned_exts:
        return fname.rsplit('.',1)[0]
    return fname.rstrip('.')
#-----------------------------------------------------------------------------
def _concatenate(listoflists):
    "Flatten a list of lists by one degree."
    finallist = []
    for sublist in listoflists: finallist.extend(sublist)
    return finallist

