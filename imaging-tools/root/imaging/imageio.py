import pylab
from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64,\
     Complex32, asarray, arange, outerproduct, ones

from odict import odict
from imaging.util import import_from

# module-private dict specifying available image readers
_readers = odict((
    ("analyze", ("imaging.analyze","readImage")),
    ("nifti", ("imaging.nifti","readImage")),
    ("fid", ("imaging.varian.FidImage","FidImage")),
    ("fdf", ("imaging.varian.FDFImage","FDFImage"))))
available_readers = _readers.keys()

# module-private dict specifying available image writers
_writers = odict((
    ("analyze", ("imaging.analyze","writeImage")),
    ("nifti", ("imaging.nifti","writeImage")),
    ("nifti_dual", ("imaging.nifti","writeImageDual"))))
available_writers = _writers.keys()


#-----------------------------------------------------------------------------
def get_dims(data):
    """
    Extract ndim, tdim, zdim, ydim, and xdim from data shape.
    @return: (ndim, tdim, zdim, ydim, xdim)
    """
    shape = data.shape
    ndim = len(shape)
    if ndim < 2 or ndim > 4:
        raise ValueError("data shape %s must be 2, 3, or 4 dimensional"%shape)
    return (ndim,) + (0,)*(4-ndim) + shape


##############################################################################
class BaseImage (object):
    """
    Interface definition for an Image.
    Attributes:
      data:  2, 3, or 4 dimensional matrix representing a slice, single
             volume, or a timecourse of volumes.
      ndim:  number of dimensions
      tdim:  number of volumes in a timecourse
      zdim:  number of slices per volume
      ydim:  number of rows per slice
      xdim:  number of columns per row
      xsize: spacial width of column
      ysize: spacial height of a row
      zsize: spacial slice thickness
      tsize: duration of each time-series volume
      x0:  position of first column
      y0:  position of first row
      z0:  position of first slice
      zRots: how many times in software image has been rotated around Z
    """

    #-------------------------------------------------------------------------
    def __init__(self, data, xsize, ysize, zsize, tsize, x0, y0, z0, zRots):
        self.setData(data)
        self.xsize, self.ysize, self.zsize, self.tsize = \
          (xsize, ysize, zsize, tsize)
        self.x0, self.y0, self.z0 = (x0, y0, z0)
        self.zRots = zRots

    #-------------------------------------------------------------------------
    def info(self):
        print "ndim =",self.ndim
        print "xdim =",self.xdim
        print "ydim =",self.ydim
        print "zdim =",self.zdim
        print "tdim =",self.tdim
        print "xsize =",self.xsize
        print "ysize =",self.ysize
        print "zsize =",self.zsize
        print "x0 =",self.x0
        print "y0 =",self.y0
        print "z0 =",self.z0
        print "data.shape =",self.data.shape
        print "data.typecode =",self.data.typecode()

    #-------------------------------------------------------------------------
    def setData(self, data):
        self.data = data
        self.ndim, self.tdim, self.zdim, self.ydim, self.xdim = get_dims(data)

    #-------------------------------------------------------------------------
    def concatenate(self, image, axis=0, newdim=False):
        self_sizes = (self.xsize, self.ysize, self.zsize)
        image_sizes = (image.xsize, image.ysize, image.zsize)

        # pixel sizes must match
        if self_sizes != image_sizes:
            raise ValueError(
              "cannot concatenate images with different pixel sizes: %s != %s"%\
              (self_sizes, image_sizes))

        newdata = newdim and asarray((self.data, image.data)) or\
                    pylab.concatenate((self.data, image.data), axis)
        return self._subimage(newdata)

    #-------------------------------------------------------------------------
    def _subimage(self, data):
        return BaseImage(data,
          self.xsize, self.ysize, self.zsize, self.tsize,
          self.x0, self.y0, self.z0, self.zRots)

    #-------------------------------------------------------------------------
    def subImage(self, subnum):
        ##!! Need to fix locations here !!##
        return self._subimage(self.data[subnum])

    #-------------------------------------------------------------------------
    def subImages(self):
        for subnum in xrange(len(self.data)): yield self.subImage(subnum)
    #-------------------------------------------------------------------------
    def zeroRots(self):
        self.zRots = 0
    #-------------------------------------------------------------------------
    def noteRot(self):
        self.zRots += 1

#-----------------------------------------------------------------------------
def get_reader(format):
    "Return an image file reader for the specified format."
    readerspec = _readers.get(format)
    if readerspec is None:
        raise ValueError("Reader '%s' not found.  Avaliable readers are: %s"%\
          (format, ", ".join(available_readers)))
    return import_from(*readerspec)

#-----------------------------------------------------------------------------
def get_writer(format):
    "Return an image file writer for the specified format."
    writerspec = _writers.get(format)
    if writerspec is None:
        raise ValueError("Writer '%s' not found.  Avaliable writers are: %s"%\
          (format, ", ".join(available_writers)))
    return import_from(*writerspec)

#-----------------------------------------------------------------------------
def readImage(filename, format, **kwargs):
    "Load an image in the specified format from the given filename."
    return get_reader(format)(filename, **kwargs)

#-----------------------------------------------------------------------------
def writeImage(image, filename, format, **kwargs):
    "Write the given image to the filesystem in the given format."
    return get_writer(format)(image, filename, **kwargs)

