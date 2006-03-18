import sys
import string 
import os
from Numeric import *
import struct
from Numeric import empty
from FFT import fft as _fft, inverse_fft as _ifft
from pylab import pi, mlab, fliplr, zeros, fromstring, angle, frange,\
  meshgrid, sqrt, exp, ones


# struct byte order constants
NATIVE = "="
LITTLE_ENDIAN = "<"
BIG_ENDIAN = ">"

def struct_format(byte_order, elements):
    return byte_order+" ".join(elements)
    
def struct_unpack(infile, byte_order, elements):
    format = struct_format(byte_order, elements)
    return struct.unpack(format, infile.read(struct.calcsize(format)))

def struct_pack(byte_order, elements, values):
    format = struct_format(byte_order, elements)
    return struct.pack(format, *values)

#-----------------------------------------------------------------------------
def import_from(modulename, objectname):
    "Import and return objectname from modulename."
    module = __import__(modulename, globals(), locals(), (objectname,))
    return getattr(module, objectname)

#-----------------------------------------------------------------------------
def shift(matrix, axis, shift):
    """
    Perform an in-place circular shift of the given matrix by the given
    number of pixels along the given axis.

    @param axis: Axis of shift: 0=x (rows), 1=y (columns), 2=z (slices), etc...
    @param shift: Number of pixels to shift.
    """
    dims = matrix.shape
    ndim = len(dims)
    if axis >= ndim: raise ValueError("bad axis %s"%axis)
    axis_dim = ndim - 1 - axis

    # construct slices
    slices = [slice(0,d) for d in dims]
    slices_new1 = list(slices)
    slices_new1[axis_dim] = slice(shift, dims[axis_dim])
    slices_old1 = list(slices)
    slices_old1[axis_dim] = slice(0, -shift)
    slices_new2 = list(slices)
    slices_new2[axis_dim] = slice(0, shift)
    slices_old2 = list(slices)
    slices_old2[axis_dim] = slice(-shift, dims[axis_dim])

    # apply slices
    new = empty(dims, matrix.typecode())
    new[tuple(slices_new1)] = matrix[tuple(slices_old1)]
    new[tuple(slices_new2)] = matrix[tuple(slices_old2)]
    matrix[:] = new

#-----------------------------------------------------------------------------
def half_shift(matrix, dim=0):
    tmp = matrix.copy()
    shift(tmp, dim, matrix.shape[-1-dim]/2)
    return tmp
    
#-----------------------------------------------------------------------------
# from image-space to k-space in FE direction (per PE line)
# in image-space: shift from (-t/2,t/2-1) to (0,t-1)
# in k-space: shift from (0,N/2) U (-(N/2-1),-w0) to (-N/2,N/2-1)
# use checkerboard masking as more efficient route
def fft(a):
    chk = checkerline(a.shape[-1])
    return chk*_fft(chk*a)

#-----------------------------------------------------------------------------
# from k-space to image-space in FE direction (per PE line)
# in k-space: shift from (-N/2,N/2-1) to (0,N/2) U (-(N/2-1),-w0)
# in image-space: shift from (0,t-1) to (-t/2,t/2-1)
# use checkerboard masking as more efficient route
def ifft(a):
    chk = checkerline(a.shape[-1])
    return chk*_ifft(chk*a)

#-----------------------------------------------------------------------------
def checkerline(cols):
    return ones(cols) - 2*(arange(cols)%2)

#-----------------------------------------------------------------------------
def checkerboard(rows, cols):
    return outerproduct(checkerline(rows), checkerline(cols))

#-----------------------------------------------------------------------------
def checkercube(slices, rows, cols):
    p = zeros((slices, rows, cols))
    q = checkerboard(rows, cols)
    for z in range(slices):
        p[z] = (1 - 2*(z%2))*q
    return p

#-----------------------------------------------------------------------------
def complex_checkerboard(rows, cols):
    return checkerboard(rows, cols) - 1.j*checkerboard(rows, cols)
 
#-----------------------------------------------------------------------------
def apply_phase_correction(image, phase):
    corrector = cos(phase) + 1.j*sin(phase)
    return fft(ifft(image)*corrector).astype(image.typecode())

#-----------------------------------------------------------------------------
def normalize_angle(a):
    "@return the given angle between -pi and pi"
    return a + where(a<-pi, 2.*pi, 0) + where(a>pi, -2.*pi, 0)

#-----------------------------------------------------------------------------
def fermi_filter(rows, cols, cutoff, trans_width):
    """
    @return: a Fermi filter kernel.
    @param cutoff: distance from the center at which the filter drops to 0.5.
      Units for cutoff are percentage of radius.
    @param trans_width: width of the transition.  Smaller values will result
      in a sharper dropoff.
    """
    row_end = (rows-1)/2.0; col_end = (cols-1)/2.0
    row_vals = frange(-row_end, row_end)**2
    col_vals = frange(-col_end, col_end)**2
    X, Y = meshgrid(row_vals, col_vals)
    return 1/(1 + exp((sqrt(X + Y) - cutoff*cols/2.0)/trans_width))

#-----------------------------------------------------------------------------
def median_filter(image, N):
    "Filter an image with a median filter of order NxN where N is odd."
    if not N%2: raise ValueError("Order of median filter must be odd.")
    ndim = len(image.shape)
    if ndim < 2:
        raise ValueError("Image dimension must be at least 2 for median_filter")
    tdim, zdim, ydim, xdim = (1,)*(4-ndim) + image.shape[-4:]

    median_pt = int((N*N)/2)
    center = int(N/2)
    image = reshape(image,(tdim*zdim, ydim, xdim))
    img = empty(image.shape, image.typecode())
    subm = empty((N, N), image.typecode())
    for tz in range(tdim*zdim):
        img[tz, 0, :] = 0.
        img[tz, -1:, :] = 0.
        for y in range(ydim-N):
            img[tz, y, 0] = 0.
            img[tz, y, xdim-1] = 0.
            for x in range(xdim-N):
                subm[:,:] = image[tz, y+1:y+N+1, x+1:x+N+1]
                s = sort(subm.flat)
                img[tz, y+center+1, x+center+1] = s[median_pt]
    return reshape(img, (tdim, zdim, ydim, xdim))

#-----------------------------------------------------------------------------
def unwrap_phase(image):
    from imaging.imageio import readImage, writeImage
    wrapped_fname = "wrapped_cmplx" 
    unwrapped_fname = "unwrapped"
    writeImage(image, wrapped_fname, "analyze")
    exec_cmd("prelude -c %s -o %s -s"%(wrapped_fname, unwrapped_fname))
    unwrapped_image = readImage(unwrapped_fname, "analyze")
    exec_cmd("/bin/rm %s.* %s.*"%(wrapped_fname, unwrapped_fname))
    return unwrapped_image

#-----------------------------------------------------------------------------
def compute_fieldmap(phase_pair, asym_time, dwell_time):
    """
    Compute fieldmap using a time-series containing two unwrapped phase
    volumes.
    """
    from imaging.imageio import readImage, writeImage
    phasepair_fname = "phasepair"
    fieldmap_fname = "fieldmap"
    writeImage(phase_pair, phasepair_fname, "analyze")
    pp = readImage(phasepair_fname, "analyze")
    pp._dump_header()
    exec_cmd("fugue -p %s --asym=%f --dwell=%f --savefmap=%s"%\
        (phasepair_fname, asym_time, dwell_time, fieldmap_fname))
    fieldmap = readImage(fieldmap_fname, "analyze")
    exec_cmd("/bin/rm %s.* %s.*"%(fieldmap_fname, phasepair_fname))
    return fieldmap

#-----------------------------------------------------------------------------
def exec_cmd(cmd, verbose=False, exit_on_error=False):
    "Execute unix command and handle errors."
    import sys, os
    if(verbose): print "\n" +  cmd
    status = os.system(cmd)
    if(status):
        print "\n****** Error occurred while executing: ******"
        print cmd
        print "Aborting procedure\n\n"
        if exit_on_error: sys.exit(1)

#-----------------------------------------------------------------------------
def resample_phase_axis(input_image,pixel_pos):
#********************************************

# Purpose: Resample along phase encode axis of epi images.

# Inputs: input_image: Epi -/volume/slice to be resampled,
#         in orientation (-/nslice//, npe, nfe)
#
#         pixel_pos: Image of resampled pixel positions,
#         in orientation (-/nslice//, npe, nfe)
#
# To be generalized into better vector operations later



    shp = input_image.shape
    ndim = len(shp)
    xdim = shp[0]
    ydim = shp[1]
    output_image = zeros((ydim,xdim)).astype(input_image.typecode())

    delta = zeros((xdim)).astype(Float)
    for y in range(ydim):
        if ndim == 1:
            vals = input_image[:]
            x = pixel_pos[:]
        elif ndim == 2:
            vals = input_image[:,y]
            x = pixel_pos[:,y]
        ix = clip(floor(x).astype(Int),0,xdim-2)
        delta = x - ix
        if ndim == 1:
            output_image[:] = ((1.-delta)*take(vals,ix) + delta*take(vals,ix+1)).astype(Float32)
        elif ndim == 2:
            output_image[:,y] = ((1.-delta)*take(vals,ix) + delta*take(vals,ix+1)).astype(Float32)
        x1 = take(vals,ix)
        x2 = take(vals,ix+1)

    return output_image



#-----------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest
    doctest.testmod()
