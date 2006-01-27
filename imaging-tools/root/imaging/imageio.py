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
      image.xsize, image.ysize, image.zsize, image.x0, image.y0, image.z0)


##############################################################################
class BaseImage (object):
    """
    Interface definition for an Image.
    Attributes:
      data:  3 or 4 dimensional matrix representing a single volume or a
        timecourse of volumes.
      ndim:  number of dimensions
      tdim:  number of volumes in a timecourse
      zdim:  number of slices per volume
      ydim:  number of rows per slice
      xdim:  number of columns per row
      xsize: spacial width of column
      ysize: spacial height of a row
      zsize: spacial slice thickness
      x0:  position of first column
      y0:  position of first row
      z0:  position of first slice
    """

    #-------------------------------------------------------------------------
    def __init__(self, data, xsize, ysize, zsize, x0, y0, z0):
        self.setData(data)
        self.xsize, self.ysize, self.zsize = (xsize, ysize, zsize)
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


#-----------------------------------------------------------------------------
def write_analyze(image, filename, datatype=None, byteswap=False):
    from imaging.analyze import AnalyzeWriter
    writer = AnalyzeWriter(image, datatype=datatype, byteswap=byteswap)
    writer.write(filename)

readers = {}
writers = {}


#-----------------------------------------------------------------------------
def _make_test_data():
    class EmptyObject:pass
    image = EmptyObject()
    tdim = 1
    zdim = 20
    ydim = 64
    xdim = 64
    image.data = randn(tdim, zdim, ydim, xdim)
    image.ndim, image.tdim, image.zdim, image.ydim, image.xdim = (3, tdim, zdim, ydim, xdim)
    image.xsize=1
    image.ysize=1
    image.zsize=1
    image.x0=1
    image.y0=1
    image.z0=1
    image.TR = image.tr =1
    image.datatype=SHORT
    image.swap=False
    return image


#-----------------------------------------------------------------------------
def _test():
    import os
    import time

    class StopWatch:
        t = 0
        def start( self ): self.t = time.time()
        def check( self ): return time.time() - self.t

    os.system( "mkdir -p scratch" )
    filename = "scratch/frame"
    image = _make_test_data()
    timer = StopWatch()

    print "Timing old write_analyze"
    from file_io import write_analyze as old_write_analyze
    timer.start()
    for _ in range(300):
        old_write_analyze( filename, image.__dict__, image.data )
    print timer.check()

    print "Timing new write_analyze"
    timer.start()
    for _ in range(300):
         write_analyze(image, filename)
    print timer.check()


#-----------------------------------------------------------------------------
if __name__ == "__main__": _test()

