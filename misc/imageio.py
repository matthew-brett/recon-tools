from Numeric import *
from MLab import fliplr, flipud
import struct

# map type string to Numeric type code
typestr2typecode = {
  "Byte": Int8,
  "Short": Int16,
  "Integer": Int32,
  "Float": Float32,
  "Double": Float64,
  "Complex": Complex32}

# max numeric range for some smaller data types
maxranges = {
  Int8:  255.,
  Int16: 32767.,
  Int32: 2147483648.}

# (numbytes, bitpix) for each type string
typelen = {
    "Byte": (2,8),
    "Short": (4,16),
    "Integer": (8,32),
    "Float": (16,32),
    "Complex": (32,64),
    "Double": (64,64)}


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
def write_analyze( filestem, header, image ):
    headername, imagename = get_analyze_filenames( filestem )
    write_analyze_header( headername, header )
    write_analyze_image( imagename, header, image )


#-----------------------------------------------------------------------------
def write_analyze_header( filename, header ):
    xdim = header['xdim']
    ydim = header['ydim']
    zdim = header['zdim']
    tdim = header['tdim']
    TR = header['TR']
    xsize = header['xsize']
    ysize = header['ysize']
    zsize = header['zsize']
    x0 = header['x0']
    y0 = header['y0']
    z0 = header['z0']

    scale_factor = header.get( "scale_factor", 1. )
    swap = header['swap']
    numbytes, bitpix = typelen.get( header['datatype'], (4,16) )
    ndim = tdim > 1 and 4 or 3

    cd = " "
    sd = " "
    hd = 0
    id = 0
    fd = 0.

    format = "%si 10s 18s i h 1s 1s 8h 4s 8s h h h h 8f f f f f f f i i 2i 80s 24s 1s 3h 4s 10s 10s 10s 10s 10s 3s i i i i 2i 2i"%\
             (swap and ">" or "<")
    
    binary_header = struct.pack( format, 348, sd, sd, id, hd, cd, cd, ndim, xdim, ydim, zdim, tdim, hd, hd, hd, sd, sd, hd, numbytes, bitpix, hd, fd, xsize, ysize, zsize, TR, fd, fd, fd, fd, scale_factor, fd, fd, fd, fd, id, id, id, id, sd, sd, cd, x0, y0, z0, sd, sd, sd, sd, sd, sd, sd, id, id, id, id, id, id, id, id)

    f = open(filename,'w')
    f.write(binary_header)
    f.close()


#-----------------------------------------------------------------------------
def write_analyze_image( filename, header, image ):
    """Write analyze image."""

    datatype = header['datatype']
    typecode = typestr2typecode[datatype]

    xdim = header['xdim']
    ydim = header['ydim']
    zdim = header['zdim']
    tdim = header['tdim']

    # Flip images left to right and top to bottom.
    image = reshape( image, (tdim, zdim, ydim, xdim) )
    newimg = zeros( (tdim,zdim,ydim,xdim) ).astype( image.typecode() )
    for t in range(tdim):
        for z in range(zdim):
            newimg[t,z,:,:] = flipud( fliplr( image[t,z,:,:] ) )

    if tdim == 1: newimg = reshape( newimg, (zdim, ydim, xdim) )

    # Make sure image values are within the range of the desired datatype
    if datatype in ("Byte", "Short", "Integer"):
        flatimg = newimg.flat
        maxval = max(abs(flatimg[argmax(flatimg)]), abs(flatimg[argmin(flatimg)]))
        if maxval == 0.: maxval = 1.e20
        maxrange = maxranges[typecode]

        # if out of desired bounds, perform scaling
        if maxval > maxrange: newimg *= (maxrange/maxval)

    # cast image values to the desired datatype
    newimg = newimg.astype( typecode )

    # perform byteswap if requested
    if header['swap']: newimg = newimg.byteswapped()

    # Write the image file.
    f = file( filename, "w" )
    f.write( newimg.tostring() )
    f.close()


#-----------------------------------------------------------------------------
def _make_test_data():
    tdim = 1
    zdim = 20
    ydim = 64
    xdim = 64
    image = ones( (tdim,zdim,ydim,xdim) ).astype( Float )
    header = {
      "tdim":tdim,
      "zdim":zdim,
      "ydim":ydim,
      "xdim":xdim,
      "xsize":1,
      "ysize":1,
      "zsize":1,
      "x0":1,
      "y0":1,
      "z0":1,
      "TR":1,
      "datatype":"Short",
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

