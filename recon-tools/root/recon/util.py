import sys
import string 
import os
import struct
# I will ditch this massive import in favor of a Numeric.foo style
from FFT import fft as _fft, inverse_fft as _ifft
from pylab import pi, zeros, frange, array, \
  meshgrid, sqrt, exp, ones, amax, floor, asarray, cumsum, putmask, diff, \
  norm, arange, empty, Int8, Int16, Int32, arange, dot, trace, cos, sin, sign,\
  putmask, take, outerproduct, where, reshape, sort, clip, Float, Float32, \
  UInt8, UInt16
from punwrap import unwrap2D


# maximum numeric range for some smaller data types
integer_ranges = {
  Int8:  127.,
  UInt8: 255.,
  Int16: 32767.,
  UInt16: 65535.,
  Int32: 2147483647.}

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
    # if it clips, then scale down
    # if it has poor integral resolution, then scale up
    if data_code in integer_ranges.keys():
        maxval = amax(abs(data).flat)
        maxrange = integer_ranges[data_code]
        scl = maxval/maxrange or 1.
        data[:] = (data/scl).astype(data_code)
    return scl

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
def reverse(seq, axis=-1): return take(seq, -1-arange(seq.shape[axis]), axis)

#-----------------------------------------------------------------------------
def embedIm(subIm, Im, yOff, xOff):
    """
    places subImage into the middle of Image, which is known to have
    dimensions twice as large as subImage (4X area)
    @param subIm: the sub-image
    @param Im: the larger image
    """
    (nSubY, nSubX) = subIm.shape
    (nY, nX) = Im.shape
    if yOff + nSubY > nY or xOff + nSubX > nX:
        print "cannot place sub-image cornered at that location"
        return
    Im[:] = zeros((nY,nX), Complex32).copy()
    Im[yOff:yOff+nSubY,xOff:xOff+nSubX] = subIm[:,:]

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
def epi_trajectory(nseg, pseq, M):
    if pseq.find('cen') > 0 or pseq.find('alt') > 0:
        if nseg > 2:
            raise NotImplementedError("centric sampling not implemented for nseg > 2")
        a = checkerline(M)
        a[:M/2] *= -1
        b = arange(M)-M/2
        b[:M/2] = abs(b[:M/2] + 1)
    else:
        a = empty(M, Int32)
        for n in range(nseg):
            a[n:M:2*nseg] = 1
            a[n+nseg:M:2*nseg] = -1
        b = floor((arange(float(M))-M/2)/float(nseg)).astype(Int32)
    return (a, b)
#-----------------------------------------------------------------------------
def apply_phase_correction(image, phase):
    "apply a phase correction to k-space"
    corrector = exp(1.j*phase)
    return fft(ifft(image)*corrector).astype(image.typecode())

#-----------------------------------------------------------------------------
def normalize_angle(a):
    "@return the given angle between -pi and pi"
    if max(abs(a)) <= pi:
        return a
    return normalize_angle(a + where(a<-pi, 2.*pi, 0) + \
                           where(a>pi, -2.*pi, 0))

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
def linReg(Y, X=None, yvar=None): 
    # find best linear line through data:
    # solve for (b,m) = (crossing, slope)
    # let sigma = 1, may use yvar for variance in the future
    if X == None: X = arange(len(Y))
    N = len(X)
    Sx = sum(X)
    Sy = sum(Y)
    Sxx = sum(X**2)
    Sxy = sum(X*Y)
    delta = N*Sxx - Sx**2
    b = (Sxx*Sy - Sx*Sxy)/delta
    m = (N*Sxy - Sx*Sy)/delta
    #res = sum((Y-(m*X+b))**2)
    res = sum(abs(Y-(m*X+b)))/float(len(X))
    return (b, m, res)

#-----------------------------------------------------------------------------
def unwrap_ref_volume(phases, fe1, fe2):
    """
    unwrap phases one "slice" at a time, where the volume
    is sliced along a single pe line (dimensions = nslice X n_fe)
    take care to move all the surfaces to roughly the same height
    @param phases is a volume of wrapped phases
    @return: uphases an unwrapped volume, shrunk to masked region
    """
    zdim,ydim,xdim = vol_shape = phases.shape
    uphases = empty(vol_shape, Float)
    zerosl, zeropt = (0, vol_shape[2]/2)

    # unwrap the volume sliced along each PE line
    # the middle of each surface should be between -pi and pi,
    # if not, put it there!
    for u in range(0,vol_shape[1],1):
        uphases[:,u,:] = unwrap2D(phases[:,u,:])
        height = uphases[zerosl,u,zeropt]
        height = int((height+sign(height)*pi)/2/pi)
        uphases[:,u,:] = uphases[:,u,:] - 2*pi*height

    return uphases[:,:,fe1:fe2]

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
# quaternion and euler rotation helpers
#-----------------------------------------------------------------------------
def qmult(a, b):
    # perform quaternion multiplication
    w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    ii = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    jj = a[0]*b[2] + a[2]*b[0] + a[3]*b[1] - a[1]*b[3]
    kk = a[0]*b[3] + a[3]*b[0] + a[1]*b[2] - a[2]*b[1]
    Q = asarray([w, ii, jj, kk])
    return Q/norm(Q)

#-----------------------------------------------------------------------------
def qconj(Q):
    return asarray([Q[0], -Q[1], -Q[2], -Q[3]])

#-----------------------------------------------------------------------------
def euler2quat(theta=0, psi=0, phi=0):
    return matrix2quat(eulerRot(phi=phi,theta=theta,psi=psi))

#-----------------------------------------------------------------------------
def matrix2quat(m):
    # m should be 3x3
    if trace(m)>0:
        S = 0.5/sqrt(1+trace(m))
        w = 0.25/S
        ii = (m[2,1]-m[1,2])*S
        jj = (m[0,2]-m[2,0])*S
        kk = (m[1,0]-m[0,1])*S
    elif (m[0,0]>m[1,1]) and (m[0,0] > m[2,2]):
        S = sqrt(1 + m[0,0] - m[1,1] - m[2,2])*2
        w = (m[2,1]-m[1,2])/S
        ii = S/4
        jj = (m[0,1] + m[1,0])/S
        kk = (m[0,2] + m[2,0])/S
    elif m[1,1] > m[2,2]:
        S = sqrt(1 + m[1,1] - m[0,0] - m[2,2])*2
        w = (m[0,2]-m[2,0])/S
        ii = (m[0,1]+m[1,0])/S
        jj = S/4
        kk = (m[1,2]+m[2,1])/S
    else:
        S = sqrt(1 + m[2,2] - m[0,0] - m[1,1])*2
        w = (m[1,0]-m[0,1])/S
        ii = (m[0,2]+m[2,0])/S
        jj = (m[1,2]+m[2,1])/S
        kk = S/4
    return asarray([w, ii, jj, kk])

#-----------------------------------------------------------------------------
def eulerRot(theta=0, psi=0, phi=0):
    # NIFTI defines A = B*C*D = Ri(theta)*Rk(psi)*Rj(phi)
    # so the quaternions will have this convention
    # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage
    aboutX = zeros((3,3),Float)
    aboutY = zeros((3,3),Float)
    aboutZ = zeros((3,3),Float)
    # bank
    aboutX[0,0] = 1
    aboutX[1,1] = aboutX[2,2] = cos(theta)
    aboutX[1,2] = sin(theta)
    aboutX[2,1] = -sin(theta)
    # attitude
    aboutY[1,1] = 1
    aboutY[0,0] = aboutY[2,2] = cos(psi)
    aboutY[0,2] = sin(psi)
    aboutY[2,0] = -sin(psi)
    # heading
    aboutZ[2,2] = 1
    aboutZ[0,0] = aboutZ[1,1] = cos(phi)
    aboutZ[0,1] = sin(phi)
    aboutZ[1,0] = -sin(phi)
    M = dot(aboutX, dot(aboutY, aboutZ))
    # make sure no rounding error proprogates from here
    putmask(M, abs(M)<1e-5, 0)
    return M

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest
    doctest.testmod()
