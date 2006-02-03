from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64, Complex32

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
    return (ndim,)+(1,)*(4-ndim)+shape

#-----------------------------------------------------------------------------
def subimage(image, data):
    return BaseImage(data,
      image.xsize, image.ysize, image.zsize, image.tsize,
      image.x0, image.y0, image.z0)


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
    """

    #-------------------------------------------------------------------------
    def __init__(self, data, xsize, ysize, zsize, tsize, x0, y0, z0):
        self.setData(data)
        self.xsize, self.ysize, self.zsize, self.tsize = \
          (xsize, ysize, zsize, tsize)
        self.x0, self.y0, self.z0 = (x0, y0, z0)

    #-------------------------------------------------------------------------
    def setData(self, data):
        self.data = data
        self.ndim, self.tdim, self.zdim, self.ydim, self.xdim = get_dims(data)

    #-------------------------------------------------------------------------
    def concatenate(image, axis=0):
        self_sizes = (self.xsize, self.ysize, self.zsize)
        image_sizes = (image.xsize, image.ysize, image.zsize)

        # pixel sizes must match
        if self_sizes != image_sizes:
            raise ValueError(
              "cannot concatenate images with different pixel sizes: %s != %s"%\
              (self_sizes, image_sizes))

        self.setData(concatenate((self.data, image.data), axis))

    #-------------------------------------------------------------------------
    def subImage(self, subnum): return subimage(self, self.data[subnum])

    #-------------------------------------------------------------------------
    def subImages(self):
        for subdata in self.data: yield subimage(self, subdata)


readers = {}
writers = {}
