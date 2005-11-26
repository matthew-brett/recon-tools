from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64, Complex32 
import struct

# maximum numeric range for some smaller data types
maxranges = {
  Int8:  255.,
  Int16: 32767.,
  Int32: 2147483648.}

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
        self.data = data
        self.ndim, self.tdim, self.zdim, self.ydim, self.xdim = get_dims(data)
        self.xsize, self.ysize, self.zsize = (xsize, ysize, zsize)
        self.x0, self.y0, self.z0 = (x0, y0, z0)

    #-------------------------------------------------------------------------
    def subImage(self, subnum): return subimage(self, self.data[subnum])

    #-------------------------------------------------------------------------
    def subImages(self):
        for subdata in self.data: yield subimage(self, subdata)


##############################################################################
class AnalyzeWriter (object):

    # (datatype, bitpix) for each Analyze datatype
    # datatype is a bit flag into a datatype byte of the Analyze header
    # bitpix is the number of bits per pixel (or voxel, as the case may be)
    BYTE = (2,8)
    SHORT = (4,16)
    INTEGER = (8,32)
    FLOAT = (16,32)
    COMPLEX = (32,64)
    DOUBLE = (64,64)

    # map Numeric typecode to Analyze datatype
    typecode2datatype = {
      Int8: BYTE,
      Int16: SHORT,
      Int32: INTEGER,
      Float32: FLOAT,
      Float64: DOUBLE,
      Complex32: COMPLEX}

    # map Analyze datatype to Numeric typecode
    datatype2typecode = dict([(v,k) for k,v in typecode2datatype.items()])

    #-------------------------------------------------------------------------
    def __init__(self, image, datatype=None, byteswap=False):
        self.image = image
        self.datatype = \
          datatype or self.typecode2datatype[image.data.typecode()]
        self.byteswap = byteswap

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
        ndim, tdim, zdim, ydim, xdim = get_dims(image.data)
        scale_factor = getattr(image, "scale_factor", 1. )
        tr = getattr(image, "tr", 0.)
        datatype, bitpix = self.datatype
        cd = sd = " "
        hd = id = 0
        fd = 0.

        format = "%si 10s 18s i h 1s 1s 8h 4s 8s h h h h 8f f f f f f f i i 2i 80s 24s 1s 3h 4s 10s 10s 10s 10s 10s 3s i i i i 2i 2i"%\
                 (self.byteswap and ">" or "<")
        # why is TR used...?
        binary_header = struct.pack(
          format, 348, sd, sd, id, hd, cd, cd, ndim, xdim, ydim, zdim, tdim,
          hd, hd, hd, sd, sd, hd, datatype, bitpix, hd, fd, image.xsize,
          image.ysize, image.zsize, tr, fd, fd, fd, fd, scale_factor,
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
        if self.datatype != self.typecode2datatype[imagedata.typecode()]:
            typecode = self.datatype2typecode[self.datatype]

            # Make sure image values are within the range of the desired datatype
            if self.datatype in (self.BYTE, self.SHORT, self.INTEGER):
                maxval = amax(abs(imagedata).flat)
                if maxval == 0.: maxval = 1.e20
                maxrange = maxranges[typecode]

                # if out of desired bounds, perform scaling
                if maxval > maxrange: imagedata *= (maxrange/maxval)

            # cast image values to the desired datatype
            imagedata = imagedata.astype( typecode )

        if self.datatype != self.COMPLEX: imagedata = abs(imagedata)

        # perform byteswap if requested
        if self.byteswap: imagedata = imagedata.byteswapped()

        # Write the image file.
        f = file( filename, "w" )
        f.write( imagedata.tostring() )
        f.close()

#-----------------------------------------------------------------------------
def write_analyze(image, filename, datatype=None, byteswap=False):
    writer = AnalyzeWriter(image, datatype=datatype, byteswap=byteswap)
    writer.write(filename)

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

