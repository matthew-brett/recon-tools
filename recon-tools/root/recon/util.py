import struct
from FFT import fft as _fft, inverse_fft as _ifft, \
     fftnd as _fftnd, inverse_fftnd as _ifftnd
from MLab import diff
import Numeric as N

from punwrap import unwrap2D

# maximum numeric range for some smaller data types
integer_ranges = {
  N.Int8:  127.,
  N.UInt8: 255.,
  N.Int16: 32767.,
  N.UInt16: 65535.,
  # even though (integer) precision is greater for (U)Int32 than Float32,
  # we would want to encode small gradations in small real-like numbers
  # in the great range of integers available.
  N.Int32: 2147483647.,
  N.UInt32: 4294967295.,
  }

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
def scale_data(data, new_typecode):
    "casts numbers in data to desired typecode in data_code"
    # if casting to an integer type, check the data range
    # if it clips, then scale down
    # if it has poor integral resolution, then scale up
    scl = 1.0
    if new_typecode in integer_ranges.keys():
        maxval = volume_max(abs(data))
        maxrange = integer_ranges[new_typecode]
        scl = maxval/maxrange or 1.
    return scl

#-----------------------------------------------------------------------------
def range_exceeds(this_dtype, that_dtype):
    a = N.array([0], this_dtype)
    b = N.array([0], that_dtype)
    # easiest condition
    if a.itemsize() < b.itemsize():
        return False
    type_list = N.typecodes.keys()
    type_list.remove('Character')
    type_list.sort()
    # this is: ['Complex', 'Float', 'Integer', 'UnsignedInteger']
    for i in range(len(type_list[1:])):
        # a Float cannot represent a Complex, nor an Int a Float, etc.
        if this_dtype in N.typecodes[type_list[i]] and \
           that_dtype in N.typecodes[type_list[i-1]]:
            return False

    # other conditions, such as a float32 representing a long int,
    # or a double float representing a float32 are allowed
    return True
    
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
    new = N.empty(dims, matrix.typecode())
    new[tuple(slices_new1)] = matrix[tuple(slices_old1)]
    new[tuple(slices_new2)] = matrix[tuple(slices_old2)]
    matrix[:] = new

#-----------------------------------------------------------------------------
def half_shift(matrix, dim=0):
    tmp = matrix.copy()
    shift(tmp, dim, matrix.shape[-1-dim]/2)
    return tmp

#-----------------------------------------------------------------------------
def reverse(seq, axis=-1): return N.take(seq, -1-N.arange(seq.shape[axis]),
                                         axis)

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
    Im[:] = N.zeros((nY,nX), subIm.typecode()).copy()
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
# make a 2D transform analogously to the 1D transform:
# ie, order the rows and columns so that (0,0) lies at (N/2,M/2)
def fft2d(a):
    chk = checkerboard(*a.shape[-2:])
    return chk*_fftnd(chk*a, axes=(-2,-1))

#-----------------------------------------------------------------------------
# make an inverse 2D transform per method above
def ifft2d(a):
    chk = checkerboard(*a.shape[-2:])
    return chk*_ifftnd(chk*a, axes=(-2,-1))

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
    return N.ones(cols) - 2*(N.arange(cols)%2)

#-----------------------------------------------------------------------------
def checkerboard(rows, cols):
    return N.outerproduct(checkerline(rows), checkerline(cols))

#-----------------------------------------------------------------------------
def checkercube(slices, rows, cols):
    p = N.zeros((slices, rows, cols))
    q = checkerboard(rows, cols)
    for z in range(slices):
        p[z] = (1 - 2*(z%2))*q
    return p

#-----------------------------------------------------------------------------
def complex_checkerboard(rows, cols):
    return checkerboard(rows, cols) - 1.j*checkerboard(rows, cols)
 
#-----------------------------------------------------------------------------
def epi_trajectory(nseg, sampling, M):
    if sampling == "centric":
        if nseg > 2:
            raise NotImplementedError("centric sampling not implemented for nseg > 2")
        a = checkerline(M)
        a[:M/2] *= -1
        b = N.arange(M)-M/2
        b[:M/2] = abs(b[:M/2] + 1)
    else:
        a = N.empty(M, N.Int32)
        for n in range(nseg):
            a[n:M:2*nseg] = 1
            a[n+nseg:M:2*nseg] = -1
        b = N.floor((N.arange(float(M))-M/2)/float(nseg)).astype(N.Int32)
    return (a, b)
#-----------------------------------------------------------------------------
def apply_phase_correction(image, phase):
    "apply a phase correction to k-space"
    corrector = N.exp(1.j*phase)
    return fft(ifft(image)*corrector).astype(image.typecode())

#-----------------------------------------------------------------------------
def normalize_angle(a):
    "@return the given angle between -pi and pi"
    if max(abs(a)) <= N.pi:
        return a
    return normalize_angle(a + N.where(a<-N.pi, 2.*N.pi, 0) + \
                           N.where(a>N.pi, -2.*N.pi, 0))

#-----------------------------------------------------------------------------
def volume_max(a):
    "return the global maximum of array a (oblsolete with numpy's a.max())"
    if len(a.shape) == 1:
        return N.maximum.reduce(a, axis=0)
    return volume_max(N.maximum.reduce(a, axis=0))
#-----------------------------------------------------------------------------
def volume_min(a):
    "return the global minimum of array a (oblsolete with numpy's a.min())"
    if len(a.shape) == 1:
        return N.minimum.reduce(a, axis=0)
    return volume_max(N.minimum.reduce(a, axis=0))
#-----------------------------------------------------------------------------
def fermi_filter(rows, cols, cutoff, trans_width):
    """
    @return: a Fermi filter kernel.
    @param cutoff: distance from the center at which the filter drops to 0.5.
      Units for cutoff are percentage of radius.
    @param trans_width: width of the transition.  Smaller values will result
      in a sharper dropoff.
    """
    row_end = rows/2.0; col_end = cols/2.0
    row_vals = N.arange(-row_end, row_end)**2
    col_vals = N.arange(-col_end, col_end)**2
    #X, Y = meshgrid(col_vals, row_vals)
    # instead of sqrt(X+Y), do this:
    dist_2d = N.sqrt(N.add.outer(row_vals, col_vals))
    return 1/(1 + N.exp((dist_2d - cutoff*cols/2.0)/trans_width))

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
    image = N.reshape(image,(tdim*zdim, ydim, xdim))
    img = N.empty(image.shape, image.typecode())
    subm = N.empty((N, N), image.typecode())
    for tz in range(tdim*zdim):
        img[tz, 0, :] = 0.
        img[tz, -1:, :] = 0.
        for y in range(ydim-N):
            img[tz, y, 0] = 0.
            img[tz, y, xdim-1] = 0.
            for x in range(xdim-N):
                subm[:,:] = image[tz, y+1:y+N+1, x+1:x+N+1]
                s = N.sort(subm.flat)
                img[tz, y+center+1, x+center+1] = s[median_pt]
    return N.reshape(img, (tdim, zdim, ydim, xdim))

#-----------------------------------------------------------------------------
def linReg(Y, X=None, yvar=None): 
    # find best linear line through data:
    # solve for (b,m) = (crossing, slope)
    # let sigma = 1, may use yvar for variance in the future
    if X == None: X = N.arange(len(Y))
    Npt = len(X)
    Sx = sum(X)
    Sy = sum(Y)
    Sxx = sum(X**2)
    Sxy = sum(X*Y)
    delta = Npt*Sxx - Sx**2
    b = (Sxx*Sy - Sx*Sxy)/delta
    m = (Npt*Sxy - Sx*Sy)/delta
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
    uphases = N.empty(vol_shape, N.Float)
    zerosl, zeropt = (vol_shape[0]/2, vol_shape[2]/2)

    # unwrap the volume sliced along each PE line
    # the middle of each surface should be between -pi and pi,
    # if not, put it there!
    for u in range(0,vol_shape[1],1):
        uphases[:,u,:] = unwrap2D(phases[:,u,:])
        height = uphases[zerosl,u,zeropt]
        height = int((height+N.sign(height)*N.pi)/2/N.pi)
        uphases[:,u,:] = uphases[:,u,:] - 2*N.pi*height

    return uphases[:,:,fe1:fe2]

#-----------------------------------------------------------------------------
### some routines from scipy ###
def mod(x,y):
    """ x - y*floor(x/y)
        For numeric arrays, x % y has the same sign as x while
        mod(x,y) has the same sign as y.
    """
    return x - y*N.floor(x*1.0/y)


#scipy's unwrap (pythonication of Matlab's routine)
def unwrap1D(p,discont=N.pi,axis=-1):
    """unwraps radian phase p by changing absolute jumps greater than
       discont to their 2*pi complement along the given axis.
    """
    p = N.asarray(p)
    nd = len(p.shape)
    dd = diff(p,axis=axis)
    slice1 = [slice(None,None)]*nd     # full slices
    slice1[axis] = slice(1,None)
    ddmod = mod(dd+N.pi,2*N.pi)-N.pi
    N.putmask(ddmod,(ddmod==-N.pi) & (dd > 0),N.pi)
    ph_correct = ddmod - dd
    N.putmask(ph_correct,abs(dd)<discont,0)
    up = N.array(p,copy=1,typecode='d')
    up[slice1] = p[slice1] + N.add.accumulate(ph_correct,axis)
    return up

#-----------------------------------------------------------------------------
def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using FFT. See convolve.
    """
    s1 = N.array(in1.shape)
    s2 = N.array(in2.shape)
    if (in1.typecode() in ['D','F']) or (in1.typecode() in ['D', 'F']):
        cmplx=1
    else: cmplx=0
    fft_size = axes and (s1[-len(axes):]+s2[-len(axes):]-1) or (s1 + s2 - 1)
    #size = s1 > s2 and s1 or s2
    IN1 = _fftnd(in1, s=fft_size, axes=axes)
    IN1 *= _fftnd(in2, s=fft_size, axes=axes)
    ret = _ifftnd(IN1, axes=axes)
    del IN1
    if not cmplx:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        osize = N.product(s1,axis=0) > N.product(s2,axis=0) and s1 or s2
        return _centered(ret,osize)
    elif mode == "valid":
        return _centered(ret,abs(s2-s1)+1)

#-----------------------------------------------------------------------------
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = N.asarray(newsize)
    currsize = N.array(arr.shape)
    startind = (currsize - newsize) / 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

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
        for sl in range(nslice):
            s = vol_data[vol, sl]
            y = pplen==4 and pixel_pos[vol, sl] or pixel_pos[sl]
            iy = N.clip(N.floor(y).astype(N.Int32), 0, n_pe-2)
            dy = y - iy
            for m in range(n_pe):
                s[:,m]=((1.-dy[:,m])*N.take(s[:,m],iy[:,m]) \
                        + dy[:,m]*N.take(s[:,m],iy[:,m]+1)).astype(N.Float32)

    return vol_data
#-----------------------------------------------------------------------------
# quaternion and euler rotation helpers
import LinearAlgebra as LA

class Quaternion:
    def __init__(self, i=0., j=0., k=0., qfac=1., M=None):
        self.Q = None
        if M is not None:
            self.matrix2quat(M)
        else:
            self.Q = N.array([i, j, k])
            self.qfac = qfac

    def matrix2quat(self, m):
        # m should be 3x3
        if len(m.shape) != 2:
            raise ValueError("Matrix must be 3x3")
            
        M = m[-3:,-3:].astype('d')

        
        xd = N.sqrt(M[0,0]**2 + M[1,0]**2 + M[2,0]**2)
        yd = N.sqrt(M[0,1]**2 + M[1,1]**2 + M[2,1]**2)
        zd = N.sqrt(M[0,2]**2 + M[1,2]**2 + M[2,2]**2)
        if xd < .0001:
            M[:,0] = N.array([1., 0., 0.])
            xd = 1.
        if yd < .0001:
            M[:,1] = N.array([0., 1., 0.])
            yd = 1.
        if zd < .0001:
            M[:,2] = N.array([0., 0., 1.])
            zd = 1.
        M[:,0] /= xd
        M[:,1] /= yd
        M[:,2] /= zd
        if N.sum((N.dot(N.transpose(M), M) - N.identity(3)).flat) > .0001:
            raise ValueError("matrix not orthogonal, must fix")

        zd = LA.determinant(M)
        if zd > 0:
            self.qfac = 1.0
        else:
            self.qfac = -1.0
            M[:,2] *= -1.0

        a = N.trace(M) + 1.0
        if a > 0.5:
            a = 0.5 * N.sqrt(a)
            b = 0.25 * (M[2,1]-M[1,2])/a
            c = 0.25 * (M[0,2]-M[2,0])/a
            d = 0.25 * (M[1,0]-M[0,1])/a
        else:
            xd = 1.0 + M[0,0] - (M[1,1] + M[2,2])
            yd = 1.0 + M[1,1] - (M[0,0] + M[2,2])
            zd = 1.0 + M[2,2] - (M[0,0] + M[1,1])
            if xd > 1.0:
                b = 0.5 * N.sqrt(xd)
                c = 0.25 * (M[0,1] + M[1,0])/b
                d = 0.25 * (M[0,2] + M[2,0])/b
                a = 0.25 * (M[2,1] - M[1,2])/b
            elif yd > 1.0:
                c = 0.5 * N.sqrt(yd)
                b = 0.25 * (M[0,1] + M[1,0])/c
                d = 0.25 * (M[1,2] + M[2,1])/c
                a = 0.25 * (M[0,2] - M[2,0])/c
            else:
                d = 0.5 * N.sqrt(zd)
                b = 0.25 * (M[0,2] + M[2,0])/d
                c = 0.25 * (M[1,2] + M[2,1])/d
                a = 0.25 * (M[1,0] - M[0,1])/d
            if a < 0.01:
                (a, b, c, d) = (-a, -b, -c, -d)

        self.Q = N.array([b, c, d])
        

    def tomatrix(self):

        R = N.empty((3,3), N.Float)
        b,c,d = tuple(self.Q)
        a = 1.0 - (b*b + c*c + d*d)
        if (a < 1.e-7):
            a = 1.0 / N.power(b*b+c*c+d*d, 0.5)
            b *= a
            c *= a
            d *= a
            a = 0. #0.001
        else:
            a = N.power(a,0.5)
        R[0,0] = (a*a + b*b - c*c - d*d)
        R[0,1] = 2.*(b*c - a*d)
        R[0,2] = 2.*(b*d + a*c)*self.qfac
        R[1,0] = 2.*(b*c + a*d)
        R[1,1] = (a*a + c*c - b*b - d*d)
        R[1,2] = 2.*(c*d - a*b)*self.qfac
        R[2,0] = 2.*(b*d - a*c)
        R[2,1] = 2.*(c*d + a*b)
        R[2,2] = (a*a + d*d - c*c - b*b)*self.qfac
        N.putmask(R, abs(R) < 1e-5, 0.)
        return R
    
    def mult(self, quat):
        return Quaternion(M=N.dot(self.tomatrix(), quat.tomatrix()))

    def __str__(self):
        return `self.Q`

    def __mul__(self, q):
        if isinstance(q, Quaternion):
            return self.mult(q)

    def __div__(self, f):
        newq = self.Q/f
        return Quaternion(w=newq[0], i=newq[1], j=newq[2], k=newq[3])

#-----------------------------------------------------------------------------
def qmult(a, b):
    # perform quaternion multiplication
    return a.mult(b)

#-----------------------------------------------------------------------------
def euler2quat(theta=0, psi=0, phi=0):
    return Quaternion(M=(eulerRot(phi=phi,theta=theta,psi=psi)))

#-----------------------------------------------------------------------------
def eulerRot(theta=0, psi=0, phi=0):
    # NIFTI defines A = B*C*D = Ri(theta)*Rk(psi)*Rj(phi)
    # so the quaternions will have this convention
    # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage
    aboutX = N.zeros((3,3),N.Float)
    aboutY = N.zeros((3,3),N.Float)
    aboutZ = N.zeros((3,3),N.Float)
    # bank
    aboutX[0,0] = 1
    aboutX[1,1] = aboutX[2,2] = N.cos(theta)
    aboutX[1,2] = N.sin(theta)
    aboutX[2,1] = -N.sin(theta)
    # attitude
    aboutY[1,1] = 1
    aboutY[0,0] = aboutY[2,2] = N.cos(psi)
    aboutY[0,2] = N.sin(psi)
    aboutY[2,0] = -N.sin(psi)
    # heading
    aboutZ[2,2] = 1
    aboutZ[0,0] = aboutZ[1,1] = N.cos(phi)
    aboutZ[0,1] = N.sin(phi)
    aboutZ[1,0] = -N.sin(phi)
    M = N.dot(aboutX, N.dot(aboutY, aboutZ))
    # make sure no rounding error proprogates from here
    N.putmask(M, abs(M)<1e-5, 0)
    return M

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest
    doctest.testmod()
