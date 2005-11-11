from pylab import amax, Int8, Int16, Int32, Float32, Float64, Complex32 
import struct

# (datatype, bitpix) for each datatype
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

# maximum numeric range for some smaller data types
maxranges = {
  Int8:  255.,
  Int16: 32767.,
  Int32: 2147483648.}

#-----------------------------------------------------------------------------
def get_analyze_filenames( filestem ):
    """
    Construct appropriate Analyze header and image file names.
    Returns (headername, imagename)
    """
    if filestem.find( ".hdr" ) != -1 or filestem.find( ".img" ) != -1:
        filestem = ".".join( filestem.split(".")[:-1] )
    return "%s.hdr"%filestem, "%s.img"%filestem

#-----------------------------------------------------------------------------
def get_dims(data):
    "Extract ndim, tdim, zdim, ydim, and xdim from data shape."
    shape = data.shape
    ndim = len(shape)
    if ndim == 3:
        tdim = 1
        zdim, ydim, xdim = shape
    elif ndim == 4:
        tdim, zdim, ydim, xdim = shape
    else:
        raise ValueError("data shape %s must be 3 or 4 dimensional"%shape)
    return ndim, tdim, zdim, ydim, xdim


##############################################################################
class AnalyzeWriter (object):

    #-------------------------------------------------------------------------
    def __init__(self, image, datatype=None, byteswap=False):
        self.image = image
        self.datatype = datatype
        self.byteswap = byteswap

    #-------------------------------------------------------------------------
    def write(self, filestem):
        "Write ANALYZE format header, image file pair."
        headername, imagename = get_analyze_filenames(filestem)
        self.write_header(headername)
        self.write_image(imagename)

    #-------------------------------------------------------------------------
    def write_header(self, filename):
        "Write ANALYZE format header (.hdr) file."
        image = self.image
        ndim, tdim, zdim, ydim, xdim = get_dims(image.imagedata)
        xsize = image.xsize
        ysize = image.ysize
        zsize = image.zsize
        x0 = image.x0
        y0 = image.y0
        z0 = image.z0
        tr = image.tr
        scale_factor = getattr(image, "scale_factor", 1. )
        datatype, bitpix = typecode2datatype[image.imagedata.typecode()]
        cd = sd = " "
        hd = id = 0
        fd = 0.

        format = "%si 10s 18s i h 1s 1s 8h 4s 8s h h h h 8f f f f f f f i i 2i 80s 24s 1s 3h 4s 10s 10s 10s 10s 10s 3s i i i i 2i 2i"%\
                 (self.byteswap and ">" or "<")
        binary_header = struct.pack(
          format, 348, sd, sd, id, hd, cd, cd, ndim, xdim, ydim, zdim, tdim,
          hd, hd, hd, sd, sd, hd, datatype, bitpix, hd, fd, xsize, ysize,
          zsize, tr, fd, fd, fd, fd, scale_factor, fd, fd, fd, fd, id, id, id,
          id, sd, sd, cd, x0, y0, z0, sd, sd, sd, sd, sd, sd, sd, id, id, id,
          id, id, id, id, id)
        f = open(filename,'w')
        f.write(binary_header)
        f.close()

    #-------------------------------------------------------------------------
    def write_image(self, filename):
        "Write ANALYZE format image (.img) file."

        imagedata = self.image.imagedata
        datatype, bitpix = typecode2datatype[imagedata.typecode()]
        ndim, tdim, zdim, ydim, xdim = get_dims(imagedata)

        needscast = self.datatype is not None and self.datatype != datatype
        if needscast:
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

        # perform byteswap if requested
        if self.byteswap: imagedata = imagedata.byteswapped()

        # Write the image file.
        f = file( filename, "w" )
        f.write( imagedata.tostring() )
        f.close()


#-----------------------------------------------------------------------------
def _make_test_data():
    tdim = 1
    zdim = 20
    ydim = 64
    xdim = 64
    image = ones( (tdim,zdim,ydim,xdim) ).astype( Float )
    header = {
      "xsize":1,
      "ysize":1,
      "zsize":1,
      "x0":1,
      "y0":1,
      "z0":1,
      "TR":1,
      "datatype":SHORT,
      "swap":False}
    return header, image


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
    header, image = _make_test_data()
    timer = StopWatch()

    print "Timing old write_analyze"
    from file_io import write_analyze as old_write_analyze
    timer.start()
    for _ in range(300):
        old_write_analyze( filename, header, image )
    print timer.check()

    print "Timing new write_analyze"
    timer.start()
    for _ in range(300):
        write_analyze( filename, header, image )
    print timer.check()


#-----------------------------------------------------------------------------
if __name__ == "__main__": _test()

