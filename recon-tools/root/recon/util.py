import struct
from numpy.fft import fft as _fft, ifft as _ifft, \
     fftn as _fftn, ifftn as _ifftn
import numpy as N

from punwrap import unwrap2D

# maximum numeric range for some smaller data types
integer_ranges = {
  N.dtype(N.int8):  N.float32(127),
  N.dtype(N.uint8): N.float32(255),
  N.dtype(N.int16): N.float32(32767),
  N.dtype(N.uint16): N.float32(65535),
  # These next max ints are special.. they are modified so that their
  # inverses are precisely specified in floating point.
  #
  # IE: 2**-31 and 1.0/(2**31 - 1) are identical in floating point:
  # '00110000000000000000000000000000'
  # This can lead to overflow errors, since (1.0 / 2**-31) = 2**31, an
  # invalid signed 32bit integer value!
  #
  # The inverses of the following numbers are created by adding the
  # least significant value into the fixed precision fraction.
  # In notation, this is
  # (1.0 + 2**-23) * 2**-31 = '00110000000000000000000000000001'
  N.dtype(N.int32): 2147483392.,
  N.dtype(N.uint32): 4294966784.,
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
def scale_data(data, new_dtype):
    "return a scaling appropriate for the new dtype"
    # if casting to an integer type, check the data range
    # if it clips, then scale down
    # if it has poor integral resolution, then scale up
    scl = 1.0
    if new_dtype in integer_ranges.keys():
        maxval = volume_max(abs(data))
        maxrange = integer_ranges[new_dtype]
        scl = (maxval/maxrange).astype(maxval.dtype) or \
              N.array([1.], maxval.dtype)
##         if new_dtype in (N.dtype(N.int32), N.dtype(N.int64)):
##             scl = scl * N.float32(1.0 + 2**-23)
    return scl

#-----------------------------------------------------------------------------
def range_exceeds(new_dtype, old_dtype):
    a = N.array([0], new_dtype)
    b = N.array([0], old_dtype)
    # easiest condition
    if a.itemsize < b.itemsize:
        return False
    type_list = ['Complex', 'Float', 'Integer', 'UnsignedInteger']
    # this is: ['Complex', 'Float', 'Integer', 'UnsignedInteger']
    for i in range(len(type_list[1:])):
        # a Float cannot represent a Complex, nor an Int a Float, etc.
        if a.dtype.char in N.typecodes[type_list[i]] and \
           b.dtype.char in N.typecodes[type_list[i-1]]:
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
    if shift==0: return
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
    new = N.empty(dims, matrix.dtype)
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
    Im[:] = N.zeros((nY,nX), subIm.dtype).copy()
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
    return chk*_fftn(chk*a, axes=(-2,-1))

#-----------------------------------------------------------------------------
# make an inverse 2D transform per method above
def ifft2d(a):
    chk = checkerboard(*a.shape[-2:])
    return chk*_ifftn(chk*a, axes=(-2,-1))

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
    return N.outer(checkerline(rows), checkerline(cols))

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
def apply_phase_correction(image, phase):
    "apply a phase correction to k-space"
    corrector = N.exp(1.j*phase)
    return fft(ifft(image)*corrector).astype(image.dtype)

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
    img = N.empty(image.shape, image.dtype)
    subm = N.empty((N, N), image.dtype)
    for tz in range(tdim*zdim):
        img[tz, 0, :] = 0.
        img[tz, -1:, :] = 0.
        for y in range(ydim-N):
            img[tz, y, 0] = 0.
            img[tz, y, xdim-1] = 0.
            for x in range(xdim-N):
                subm[:,:] = image[tz, y+1:y+N+1, x+1:x+N+1]
                s = N.sort(subm.flatten())
                img[tz, y+center+1, x+center+1] = s[median_pt]
    return N.reshape(img, (tdim, zdim, ydim, xdim))

#-----------------------------------------------------------------------------
def linReg(Y, X=None, sigma=None, axis=-1): 
    # find best linear line through data:
    # solve for (b,m,res) = (crossing, slope, residual from fit)
    # if sigma is None, let sigma = 1 for every y-point

    if axis != -1:
        Y = N.swapaxes(Y, axis, -1)
        if X is not None:
            X = N.swapaxes(X, axis, -1)
        if sigma is not None:
            sigma = N.swapaxes(sigma, axis, -1)

    Yshape = Y.shape
    nrows = N.product(Yshape[:-1])
    npts = Yshape[-1]
    # make this 2D so it's easier to think about
    Y = N.reshape(Y, (nrows, npts))
    if X is None:
        # if no X provided, get the right number of rows of [0,1,2,...,N]
        X = N.outer(N.ones((nrows,)), N.arange(npts))
    if sigma is None:
        sigma = N.ones(Y.shape)
    S = N.power(sigma, -1.0).sum(axis=-1)
    Sx = (X/sigma).sum(axis=-1)
    Sy = (Y/sigma).sum(axis=-1)
    Sxx = (N.power(X,2)/sigma).sum(axis=-1)
    Sxy = (X*Y/sigma).sum(axis=-1)
    delta = S*Sxx - N.power(Sx,2)
    b = (Sxx*Sy - Sx*Sxy)/delta
    m = (S*Sxy - Sx*Sy)/delta
    res = abs(Y - (m[:,N.newaxis]*X+b[:,N.newaxis])).sum(axis=-1)/float(npts)

    Y = N.reshape(Y, Yshape)
    if axis != -1:
        Y = N.swapaxes(Y, axis, -1)
        # swap these back too, even if they're just local
        X = N.swapaxes(X, axis, -1)
        sigma = N.swapaxes(sigma, axis, -1)
    return (N.reshape(b, Y.shape[:-1]),
            N.reshape(m, Y.shape[:-1]),
            N.reshape(res, Y.shape[:-1]))

#-----------------------------------------------------------------------------
def polyfit(x, y, deg, sigma=None, rcond=None, full=False):
    """Least squares polynomial fit.

    Yoinked from numpy and modified to minimize with respect to variance.

    sigma kwarg should be VARIANCE and NOT STDEV

    Required arguments

        x -- vector of sample points
        y -- vector or 2D array of values to fit
        deg -- degree of the fitting polynomial

    Keyword arguments

        rcond -- relative condition number of the fit (default len(x)*eps)
        full -- return full diagnostic output (default False)

    Returns

        full == False -- coefficients
        full == True -- coefficients, residuals, rank, singular values, rcond.

    Warns

        RankWarning -- if rank is reduced and not full output

    Do a best fit polynomial of degree 'deg' of 'x' to 'y'.  Return value is a
    vector of polynomial coefficients [pk ... p1 p0].  Eg, for n=2

      p2*x0^2 +  p1*x0 + p0 = y1
      p2*x1^2 +  p1*x1 + p0 = y1
      p2*x2^2 +  p1*x2 + p0 = y2
      .....
      p2*xk^2 +  p1*xk + p0 = yk


    Method: if X is a the Vandermonde Matrix computed from x (see
    http://mathworld.wolfram.com/VandermondeMatrix.html), then the
    polynomial least squares solution is given by the 'p' in

      X*p = y

    where X is a len(x) x N+1 matrix, p is a N+1 length vector, and y
    is a len(x) x 1 vector

    This equation can be solved as

      p = (XT*X)^-1 * XT * y

    where XT is the transpose of X and -1 denotes the inverse. However, this
    method is susceptible to rounding errors and generally the singular value
    decomposition is preferred and that is the method used here. The singular
    value method takes a paramenter, 'rcond', which sets a limit on the
    relative size of the smallest singular value to be used in solving the
    equation. This may result in lowering the rank of the Vandermonde matrix,
    in which case a RankWarning is issued. If polyfit issues a RankWarning, try
    a fit of lower degree or replace x by x - x.mean(), both of which will
    generally improve the condition number. The routine already normalizes the
    vector x by its maximum absolute value to help in this regard. The rcond
    parameter may also be set to a value smaller than its default, but this may
    result in bad fits. The current default value of rcond is len(x)*eps, where
    eps is the relative precision of the floating type being used, generally
    around 1e-7 and 2e-16 for IEEE single and double precision respectively.
    This value of rcond is fairly conservative but works pretty well when x -
    x.mean() is used in place of x.

    The warnings can be turned off by:

    >>> import numpy
    >>> import warnings
    >>> warnings.simplefilter('ignore',numpy.RankWarning)

    DISCLAIMER: Power series fits are full of pitfalls for the unwary once the
    degree of the fit becomes large or the interval of sample points is badly
    centered. The basic problem is that the powers x**n are generally a poor
    basis for the functions on the sample interval with the result that the
    Vandermonde matrix is ill conditioned and computation of the polynomial
    values is sensitive to coefficient error. The quality of the resulting fit
    should be checked against the data whenever the condition number is large,
    as the quality of polynomial fits *can not* be taken for granted. If all
    you want to do is draw a smooth curve through the y values and polyfit is
    not doing the job, try centering the sample range or look into
    scipy.interpolate, which includes some nice spline fitting functions that
    may be of use.

    For more info, see
    http://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html,
    but note that the k's and n's in the superscripts and subscripts
    on that page.  The linear algebra is correct, however.

    See also polyval

    """
    order = int(deg) + 1
    if sigma is None:
        sigma = N.ones(y.shape[-1])
    inv_stdev = N.power(sigma, -0.5)
    x = N.asarray(x) + 0.0
    y = (N.asarray(y) + 0.0)*inv_stdev

    # check arguments.
    if deg < 0 :
        raise ValueError, "expected deg >= 0"
    if x.ndim != 1 or x.size == 0:
        raise TypeError, "expected non-empty vector for x"
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError, "expected 1D or 2D array for y"
    if x.shape[0] != y.shape[0] :
        raise TypeError, "expected x and y to have same length"

    # set rcond
    if rcond is None :
        xtype = x.dtype
        if xtype == N.single or xtype == N.csingle :
            rcond = len(x)*N.finfo(N.single).eps
        else :
            rcond = len(x)*N.finfo(N.double).eps

    # scale x to improve condition number
    scale = abs(x).max()
    if scale != 0 :
        x /= scale

    # solve least squares equation for powers of x
    v = N.vander(x, order)*inv_stdev[:,None]
    c, resids, rank, s = N.linalg.lstsq(v, y, rcond)

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        import warnings
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    # scale returned coefficients
    if scale != 0 :
        c /= N.vander([scale], order)[0]

    if full :
        return c, resids, rank, s, rcond
    else :
        return c

#-----------------------------------------------------------------------------
def bivariate_fit(z, dim0, dim1, deg, sigma=None, mask=None,
                  rcond=None, scale=None):
    # Want to solve this problem:
    # z[l,m] = sum_q,r (a[q,r]*dim0[l]^q*dim1[m]^r)
    #
    # If dim0 is L points and dim1 is M points, reorder data into an P point
    # vector, where P = LM (p = l*L + m)
    #
    # also, reorder the system of equations into K = QR (k = q*Q + r) equations
    #
    # Now the set of equations is z[p] = sum_k (a[k]*Xk(x[p]))
    # where a[k] -> a[q,r] and Xk[x[p]] = dim0[l]^q * dim1[m]^r
    #
    # So design matrix A will be structured:
    # | X0[dim00,dim10] X1[dim00,dim10] X2[dim00,dim10] ... XQR[dim00,dim10] |
    # | X0[dim00,dim11] X1[dim00,dim11] X2[dim00,dim11] ... XQR[dim00,dim11] |
    # |  .         .         .             .         |
    # |  .         .         .             .         |    
    # | X0[dim0L,dim1M] X1[dim0L,dim1M] X2[dim0L,dim1M] ... XQR[dim0L,dim1M] |
    #

    # a little helper function to break vector or number L into two indices
    def decompose(l,L):
        return N.floor(l/L).astype(N.int32), (l%L).astype(N.int32)
    
    dim0 = N.asarray(dim0) + 0.0
    dim1 = N.asarray(dim1) + 0.0
    z = N.asarray(z) + 0.0
    L,M = dim0.shape[0],dim1.shape[0]
    if N.product(z.shape) != L*M:
        raise ValueError("dimensions of data vector don't match the grid dimensions")
    Q = R = int(deg) + 1

    # reorder the data, variance, and mask arrays into L*M length vectors
    if sigma is None:
        sigma = N.ones((L,M))
    inv_stdev = N.power(N.reshape(sigma, (L*M,)), -0.5)

    if mask is None:
        mask = N.ones(z.shape)
    unmasked = (N.reshape(mask, (L*M,))).nonzero()[0]

    z = N.reshape(z, (L*M,))*inv_stdev
    l, m = decompose(N.arange(L*M), M)
    q, r = decompose(N.arange(Q*R), R)

    # set rcond
    if rcond is None :
        ztype = z.dtype
        if ztype == N.single or ztype == N.csingle :
            rcond = L*M*N.finfo(N.single).eps
        else :
            rcond = L*M*N.finfo(N.double).eps

    # design matrix
    if not scale:
        scale = abs(dim0).max(), abs(dim1).max()

    dim0 /= scale[0]
    dim1 /= scale[1]
    
    A = N.power(N.outer(dim0[l], N.ones(Q*R)), q) * \
        N.power(N.outer(dim1[m], N.ones(Q*R)), r)

    A_calc = N.take(A*inv_stdev[:,None], unmasked, axis=0)

    c, resids, rank, s = N.linalg.lstsq(A_calc, N.take(z, unmasked), rcond)

    vec_scale = N.power(scale[0], q)*N.power(scale[1],r)
    c /= vec_scale

    return A*vec_scale, c
    

#-----------------------------------------------------------------------------
def unwrap_ref_volume(vol, fe1, fe2):
    """
    unwrap phases one "slice" at a time, where the volume
    is sliced along a single pe line (dimensions = nslice X n_fe)
    take care to move all the surfaces to roughly the same height
    @param phases is a volume of wrapped phases
    @return: uphases an unwrapped volume, shrunk to masked region
    """
    zdim,ydim,xdim = vol_shape = vol.shape
    uphases = N.empty(vol_shape, N.float64)
    zeropt = vol_shape[2]/2
    # search for best sl to be s=0 by looking at all s where q2=0,q1=0
    scut = abs(vol[:,vol_shape[1]/2,zeropt])
    zerosl = N.nonzero(scut == scut.max())[0][0]
    phases = N.angle(vol)
    # unwrap the volume sliced along each PE line
    # the middle of each surface should be between -pi and pi,
    # if not, put it there!
    for u in range(0,vol_shape[1],1):
        uphases[:,u,:] = unwrap2D(phases[:,u,:])
        height = uphases[zerosl,u,zeropt]
        height = int((height+N.sign(height)*N.pi)/2/N.pi)
        uphases[:,u,:] = uphases[:,u,:] - 2*N.pi*height

    #heights = N.zeros(ydim)
    #heights[1:] = unwrap1D(uphases[zerosl,:,zeropt], return_diffs=True)
    #uphases[:] = uphases + heights[N.newaxis,:,N.newaxis]
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
def unwrap1D(p,discont=N.pi,axis=-1,return_diffs=False):
    """unwraps radian phase p by changing absolute jumps greater than
       discont to their 2*pi complement along the given axis.
    """
    p = N.asarray(p)
    nd = len(p.shape)
    dd = N.diff(p,axis=axis)
    slice1 = [slice(None,None)]*nd     # full slices
    slice1[axis] = slice(1,None)
    ddmod = mod(dd+N.pi,2*N.pi)-N.pi
    N.putmask(ddmod,(ddmod==-N.pi) & (dd > 0),N.pi)
    ph_correct = ddmod - dd
    N.putmask(ph_correct,abs(dd)<discont,0)
    up = N.array(p,copy=1, dtype=N.float64)
    up[slice1] = p[slice1] + N.add.accumulate(ph_correct,axis)
    if not return_diffs:
        return up
    else:
        return N.add.accumulate(ph_correct,axis)

#-----------------------------------------------------------------------------
def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using FFT. See convolve.
    """
    s1 = N.array(in1.shape)
    s2 = N.array(in2.shape)
    if (in1.dtype.char in ['D','F']) or (in2.dtype.char in ['D', 'F']):
        cmplx=1
    else: cmplx=0
    if axes:
        fft_size = (s1[-len(axes):]+s2[-len(axes):]-1)
    else:
        fft_size = (s1 + s2 - 1)
    #size = s1 > s2 and s1 or s2
    IN1 = _fftn(in1, s=fft_size, axes=axes)
    IN1 *= _fftn(in2, s=fft_size, axes=axes)
    ret = _ifftn(IN1, axes=axes)
    del IN1
    if not cmplx:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if N.product(s1, axis=0) > N.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
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
            iy = N.clip(N.floor(y).astype(N.int32), 0, n_pe-2)
            dy = y - iy
            for m in range(n_pe):
                s[:,m]=((1.-dy[:,m])*N.take(s[:,m],iy[:,m]) \
                        + dy[:,m]*N.take(s[:,m],iy[:,m]+1)).astype(N.float32)

    return vol_data
#-----------------------------------------------------------------------------
# quaternion and euler rotation helpers
#import LinearAlgebra as LA

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

        if (N.dot(N.transpose(M), M) - N.identity(3)).sum() > .0001:
            raise ValueError("matrix not orthogonal, must fix")

        zd = N.linalg.det(M)
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

        R = N.empty((3,3), N.float64)
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
        N.putmask(R, abs(R) < 1e-5, 0)
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
    aboutX = N.zeros((3,3),N.float)
    aboutY = N.zeros((3,3),N.float64)
    aboutZ = N.zeros((3,3),N.float64)
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
