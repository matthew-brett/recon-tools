import Numeric as N

from odict import odict
from recon.util import import_from, Quaternion

# module-private dict specifying available image readers
_readers = odict((
    ("analyze", ("recon.analyze","readImage")),
    ("nifti", ("recon.nifti","readImage")),
    ("fid", ("recon.scanners.varian.FidImage","FidImage")),
    ("fdf", ("recon.scanners.varian.FDFImage","FDFImage"))))
available_readers = _readers.keys()

# module-private dict specifying available image writers
_writers = odict((
    ("analyze", ("recon.analyze","AnalyzeWriter")),
    ("nifti-single", ("recon.nifti","NiftiWriter")),
    ("nifti-dual", ("recon.nifti","NiftiWriter"))))
available_writers = _writers.keys()

#-----------------------------------------------------------------------------
def get_dims(data):
    """
    Extract ndim, tdim, zdim, ydim, and xdim from data shape.
    @return: (ndim, tdim, zdim, ydim, xdim)
    """
    # For consistency, 1-volume images have 4 data dimensions.
    # Let's not reflect this in the image properties. 
    shape = data.shape[0] == 1 and data.shape[1:] or data.shape
    ndim = len(shape)
    if ndim < 2 or ndim > 4:
        raise ValueError("data shape %s must be 2, 3, or 4 dimensional"%shape)
    return (ndim,) + (0,)*(4-ndim) + shape

##############################################################################
class DataChunk (object):
    def __init__(self, data, num):
        self._data = data
        self.num = num
        self.shape = data.shape

    def __getitem__(self, slicer):
        return self._data[slicer]

    def __setitem__(self, slicer, data):
        self._data[slicer] = N.asarray(data).astype(self._data.typecode())
    
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
      zdim:  number of slices per volume
      ydim:  number of rows per slice
      xdim:  number of columns per row
      xsize: spacial width of column
      ysize: spacial height of a row
      zsize: spacial slice thickness
      tsize: duration of each time-series volume
      x0: x coordinate of voxel-origin ??
      y0: y coordinate of voxel-origin ?? 
      z0: z coordinate of voxle-origin ??
      orientation: name of the orientaion (coronal, axial, etc)
      orientation_xform: quaternion describing the orientation

    capabilities provided:
      volume/slice slicing
      fe/pe slicing
      __getitem__, __setitem__
      data xform (abs, real, imag, etc)
    """

    #-------------------------------------------------------------------------
    def __init__(self, data, xsize, ysize, zsize, tsize,
                 origin=None, orient_xform=None, orient_name=None):
        self.setData(data)
        self.xsize, self.ysize, self.zsize, self.tsize = \
          (xsize, ysize, zsize, tsize)
        self.x0, self.y0, self.z0 = origin or (self.xsize*self.xdim/2.,
                                               self.ysize*self.ydim/2.,
                                               self.zsize*self.zdim/2.)
        self.orientation_xform = orient_xform or Quaternion()
        self.orientation = orient_name or ""

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
        self.shape = (self.tdim and (self.tdim,) or ()) + \
                     (self.zdim, self.ydim, self.xdim)

    #-------------------------------------------------------------------------
    def concatenate(self, image, axis=0, newdim=False):
        self_sizes = (self.xsize, self.ysize, self.zsize)
        image_sizes = (image.xsize, image.ysize, image.zsize)

        # pixel sizes must match
        if self_sizes != image_sizes:
            raise ValueError(
              "cannot concatenate images with different pixel sizes: %s != %s"%\
              (self_sizes, image_sizes))

        newdata = newdim and N.asarray((self.data, image.data)) or \
                  N.concatenate((self.data, image.data), axis)
        return self._subimage(newdata)

    #-------------------------------------------------------------------------
    def __iter__(self):
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
        return self.data[slicer]
    #-------------------------------------------------------------------------
    def __setitem__(self, slicer, newdata):
        ndata = N.asarray(newdata)
        if ndata.typecode().isupper() and self.data.typecode().islower():
            print "warning: losing information on complex->real cast!"
        self.data[slicer] = ndata.astype(self.data.typecode())
    #-------------------------------------------------------------------------
    def __mul__(self, a):
        self[:] = self[:]*a
    #-------------------------------------------------------------------------
    def __div__(self, a):
        self[:] = self[:]/a
    #-------------------------------------------------------------------------
    def _subimage(self, data):
        return ReconImage(data,
                          self.xsize, self.ysize, self.zsize, self.tsize,
                          origin=(self.x0, self.y0, self.z0),
                          orient_xform=self.orientation_xform,
                          orient_name=self.orientation)

    #-------------------------------------------------------------------------
    def subImage(self, subnum):
        ##!! Need to fix locations here !!##
        return self._subimage(self.data[subnum])

    #-------------------------------------------------------------------------
    def subImages(self):
        for subnum in xrange(len(self.data)): yield self.subImage(subnum)

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
def clean_name(fname):
    pruned_exts = ['nii', 'hdr', 'img']
    if fname.rsplit('.')[-1] in pruned_exts:
        return fname.rsplit('.',1)[0]
    return fname
    
#-----------------------------------------------------------------------------
def readImage(filename, format, **kwargs):
    "Load an image in the specified format from the given filename."
    return get_reader(format)(clean_name(filename), **kwargs)

#-----------------------------------------------------------------------------
def _concatenate(listoflists):
    "Flatten a list of lists by one degree."
    finallist = []
    for sublist in listoflists: finallist.extend(sublist)
    return finallist

#-----------------------------------------------------------------------------
## def writeImage(image, filename, format, **kwargs):
##     "Write the given image to the filesystem in the given format."
##     return get_writer(format)(image, clean_name(filename), **kwargs)

def writeImage(image, filestem, datatype=None, targetdim=None,
               filetype=None, suffix=None, scale=1.0, format="analyze"):
    Writer = get_writer(format)
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
        Writer(subimage, datatype=datatype,
               filetype=filetype, scale=scale).write(substem)
