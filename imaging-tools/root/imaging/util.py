import sys
import string 
import os
from Numeric import *
import struct
from Numeric import empty
from FFT import fft as _fft, inverse_fft as _ifft
from pylab import pi, mlab, fliplr, zeros, fromstring, angle, frange,\
  meshgrid, sqrt, exp, ones, amax, floor, asarray, cumsum, putmask, diff


# maximum numeric range for some smaller data types
maxranges = {
  Int8:  255.,
  Int16: 32767.,
  Int32: 2147483648.}

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
def castData(data, data_code):
    "casts numbers in data to desired typecode in data_code"
    # if casting to an integer type, check the data range
    if data_code in (Int8, Int16, Int32):
        maxval = amax(abs(data).flat)
        if maxval == 0.: maxval = 1.e20
        maxrange = maxranges[data_code]
        if maxval > maxrange: data *= (maxrange/maxval)
    # make the cast
    data = data.astype(data_code)
    
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
    X, Y = meshgrid(col_vals, row_vals)
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
def linReg(X, Y, yvar=None): 
    # find best linear line through data:
    # solve for (b,m) = (crossing, slope)
    # let sigma = 1, may use yvar for variance in the future
    N = len(X)
    Sx = sum(X)
    Sy = sum(Y)
    Sxx = sum(X**2)
    Sxy = sum(X*Y)
    delta = N*Sxx - Sx**2
    b = (Sxx*Sy - Sx*Sxy)/delta
    m = (N*Sxy - Sx*Sy)/delta
    res = sum((Y-(m*X+b))**2)
    return (b, m, res)

#-----------------------------------------------------------------------------
### some routines from scipy ###
def mod(x,y):
    """ x - y*floor(x/y)
        For numeric arrays, x % y has the same sign as x while
        mod(x,y) has the same sign as y.
    """
    return x - y*floor(x*1.0/y)


#scipy's unwrap (pythonication of Matlab's routine)
def unwrap1D(p,discont=pi,axis=-1):
    """unwraps radian phase p by changing absolute jumps greater than
       discont to their 2*pi complement along the given axis.
    """
    p = asarray(p)
    nd = len(p.shape)
    dd = diff(p,axis=axis)
    slice1 = [slice(None,None)]*nd     # full slices
    slice1[axis] = slice(1,None)
    ddmod = mod(dd+pi,2*pi)-pi
    putmask(ddmod,(ddmod==-pi) & (dd > 0),pi)
    ph_correct = ddmod - dd;
    putmask(ph_correct,abs(dd)<discont,0)
    up = array(p,copy=1,typecode='d')
    up[slice1] = p[slice1] + cumsum(ph_correct,axis)
    return up

#-----------------------------------------------------------------------------
### alternate definition of unwrap1D
## def unwrap1D(p):
##     if len(p.shape) < 2:
##         p = reshape(p, (1, p.shape[0]))
##     dd = diff(p)
##     dd_wr = arctan2(sin(dd), cos(dd))
##     uph = zeros(p.shape, Float)
##     uph[:,0] = p[:,0]
##     for col in range(dd.shape[-1]):
##         uph[:,col+1] = uph[:,col] + dd_wr[:,col]
##     return uph
#-----------------------------------------------------------------------------
## "Depricated" 4/24/06
## def unwrap_phase(image):
##     from imaging.imageio import readImage, writeImage
##     wrapped_fname = "wrapped_cmplx" 
##     unwrapped_fname = "unwrapped"
##     writeImage(image, wrapped_fname, "analyze")
##     exec_cmd("prelude -c %s -o %s"%(wrapped_fname, unwrapped_fname))
##     unwrapped_image = readImage(unwrapped_fname, "analyze")
##     exec_cmd("/bin/rm %s.* %s.*"%(wrapped_fname, unwrapped_fname))
##     return unwrapped_image

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
def resample_phase_axis(vol_data, pixel_pos):
    """Purpose: Resample along phase encode axis of epi images.
    Performs a trilinear interpolation along the y (pe) axis. This
    reduces to:
    V[x,y,z] = interp(V, x, y+phasemap, z) = (1-dy)*v000 + dy*v010
    where dy = y+phasemap - floor(y+phasemap) = (y' - iy')
    v000 = V[x,iy',z]
    v010 = V[z,iy'+1,z]
    
    @param vol_data: Epi volume to be resampled, in orientation
    (nvol, nslice, npe, nfe)

    @param pixel_pos: Image of resampled pixel positions,
    in orientation (nslice, npe, nfe)

    @return: volume resampled along phase axis
    
    Warning: beware of any rotations which would switch expected axes!"""

    vdshape, ppshape = vol_data.shape, pixel_pos.shape
    vdlen, pplen = len(vdshape), len(ppshape)

    if vdlen < 3 or pplen < 3:
        print "needs at least 3 dimensions for both args, returning"
        return
       
    if vdshape[-3:] != ppshape[-3:]:
        print "phase map dimensions don't match volume dimensions, returning"
        return

    nvol = (vdlen < 4) and 1 or vdshape[0]  # vdlen should always be 4?)
    nslice, n_pe, n_fe =  vdshape[1:]

    for vol in range(nvol):
        for slice in range(nslice):
            s = vol_data[vol, slice]
            y = pplen==4 and pixel_pos[vol, slice] or pixel_pos[slice]
            iy = clip(floor(y).astype(Int), 0, n_pe-2)
            dy = y - iy
            for m in range(n_pe):
                s[:,m]=((1.-dy[:,m])*take(s[:,m],iy[:,m]) \
                        + dy[:,m]*take(s[:,m],iy[:,m]+1)).astype(Float32)

    return vol_data


#-----------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest
    doctest.testmod()
