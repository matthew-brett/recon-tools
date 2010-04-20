import struct
import numpy as np
from punwrap import unwrap3D, unwrap2D
from fftmod import fft1, ifft1, fft2, ifft2
from numpy.fft import fftn as _fftn, ifftn as _ifftn
from recon import tempfile, loads_extension_on_call

# maximum numeric range for some smaller data types
integer_ranges = {
  np.dtype(np.int8):  np.float32(127),
  np.dtype(np.uint8): np.float32(255),
  np.dtype(np.int16): np.float32(32767),
  np.dtype(np.uint16): np.float32(65535),
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
  np.dtype(np.int32): np.float32(2147483392),
  np.dtype(np.uint32): np.float32(4294966784),
  }

#-----------------------------------------------------------------------------
def scale_data(data, new_dtype):
    "return a scaling appropriate for the new dtype"
    # if casting to an integer type, check the data range
    # if it clips, then scale down
    # if it has poor integral resolution, then scale up
    scl = 1.0
    if new_dtype in integer_ranges.keys():
        maxval = abs(data).max()
        maxrange = integer_ranges[new_dtype]
        scl = (maxval/maxrange).astype(maxval.dtype) or \
              np.array([1.], maxval.dtype)
    return scl

#-----------------------------------------------------------------------------
def range_exceeds(new_dtype, old_dtype):
    a = np.array([0], new_dtype)
    b = np.array([0], old_dtype)
    # easiest condition
    if a.itemsize < b.itemsize:
        return False
    type_list = ['Complex', 'Float', 'Integer', 'UnsignedInteger']
    for i in range(len(type_list[1:])):
        # a Float cannot represent a Complex, nor an Int a Float, etc.
        if a.dtype.char in np.typecodes[type_list[i]] and \
           b.dtype.char in np.typecodes[type_list[i-1]]:
            return False

    # other conditions, such as a float32 representing a long int,
    # or a double float representing a float32 are allowed
    return True
    
#-----------------------------------------------------------------------------
def shift(matrix, shift, axis=-1):
    """
    Perform a circular shift of the given matrix by the given
    number of pixels along the given axis.
    Leaves matrix[n] := matrix[n-shift]

    @param axis: Axis of shift
    @param shift: Number of pixels to shift.
    """
    dims = matrix.shape
    ndim = len(dims)
    while axis < 0:
        axis += ndim
    if axis >= ndim: raise ValueError("bad axis %s"%axis)
    if shift==0: return

    # construct slices
    slices = [slice(0,d) for d in dims]
    slices_new1 = list(slices)
    slices_new1[axis] = slice(shift, dims[axis])
    slices_old1 = list(slices)
    slices_old1[axis] = slice(0, -shift)
    slices_new2 = list(slices)
    slices_new2[axis] = slice(0, shift)
    slices_old2 = list(slices)
    slices_old2[axis] = slice(-shift, dims[axis])

    # apply slices
    new = np.empty(dims, matrix.dtype)
    new[tuple(slices_new1)] = matrix[tuple(slices_old1)]
    new[tuple(slices_new2)] = matrix[tuple(slices_old2)]
    matrix[:] = new

#-----------------------------------------------------------------------------
def embedIm(sub_img, img, y_off, x_off):
    """
    places a sub-image into a larger image, which should have dimensions
    at least as large as (sub_y+y_off, sub_x+x_off)
    @param sub_img: the sub-image
    @param img: the larger image
    """
    (sub_y, sub_x) = sub_img.shape[-2:]
    (y, x) = img.shape[-2:]
    if y_off + sub_y > y or x_off + sub_x > x:
        print "cannot place sub-image cornered at that location"
        return
    img[:] = np.zeros((y,x), sub_img.dtype)
    img[y_off:y_off+sub_y, x_off:x_off+sub_x] = sub_img[:,:]

#-----------------------------------------------------------------------------
def fft(a):
    "A light wrapper of the fftmod's 1D fft, see fftmod doc for more info."
    return fft1(a)
#-----------------------------------------------------------------------------
def ifft(a):
    "A light wrapper of the fftmod's 1D ifft, see fftmod doc for more info."    
    return ifft1(a)
#-----------------------------------------------------------------------------
def fft2d(a):
    "A light wrapper of the fftmod's 2D fft, see fftmod doc for more info."
    return fft2(a)
#-----------------------------------------------------------------------------
def ifft2d(a):
    "A light wrapper of the fftmod's 2D ifft, see fftmod doc for more info."    
    return ifft2(a)
#-----------------------------------------------------------------------------
def checkerline(cols):
    "Returns a 1D array of (-1)^n"
    return np.ones(cols) - 2*(np.arange(cols)%2)
#-----------------------------------------------------------------------------
def checkerboard(rows, cols):
    "Returns a 2D array of (-1)^(n+m)"
    return np.outer(checkerline(rows), checkerline(cols))
#-----------------------------------------------------------------------------
def checkercube(slices, rows, cols):
    "Returns a 3D array of (-1)^(n+m+r)"
    p = np.zeros((slices, rows, cols))
    q = checkerboard(rows, cols)
    for z in range(slices):
        p[z] = (1 - 2*(z%2))*q
    return p
#-----------------------------------------------------------------------------
def apply_phase_correction(image, phase):
    """
    Applies a phase correction array to a k-space array, taking care
    to transform and multiply in-place
    """
    ifft1(image, inplace=True)
    np.multiply(image, phase, image)
    fft1(image, inplace=True)
#-----------------------------------------------------------------------------
def normalize_angle(a):
    "Returns the given angle wrapped between -pi and pi"
    if np.abs(a).max() <= np.pi:
        return a
    return normalize_angle(a + np.where(a<-np.pi, 2.*np.pi, 0) + \
                           np.where(a>np.pi, -2.*np.pi, 0))

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
    row_vals = np.arange(-row_end, row_end)**2
    col_vals = np.arange(-col_end, col_end)**2
    dist_2d = np.sqrt(np.add.outer(row_vals, col_vals))
    return 1/(1 + np.exp((dist_2d - cutoff*cols/2.0)/trans_width))

#-----------------------------------------------------------------------------
def export_extension(build=False):

    from scipy.weave import ext_tools
    from os.path import join, split

    code = \
"""
PyArrayIterObject *yitr, *xitr, *sitr, *mskitr, *mitr, *bitr, *chitr;
double *mptr, *bptr, *chiptr;
double sx, sy, s, sxx, sxy, delta, diff;
int k, p;
int ylen, ystride, sstride, xstride, mstride, un_count;
int *unmasked;

ytype *yptr, y1;
xtype *xptr, x1;
stype *sptr, sig;
mtype *mskptr; 

yitr = (PyArrayIterObject *) PyArray_IterAllButAxis(py_y, &caxis);
mitr = (PyArrayIterObject *) PyArray_IterNew(py_m);
bitr = (PyArrayIterObject *) PyArray_IterNew(py_b);
chitr = (PyArrayIterObject *) PyArray_IterNew(py_chisq);

ylen = Ny[caxis];
if(has_x)
  xitr = (PyArrayIterObject *) PyArray_IterAllButAxis(py_x, &caxis);
else {
  xptr = new xtype[ylen];
  for(k=0; k<ylen; k++) xptr[k] = (xtype) k;
}
if(has_sigma)
  sitr = (PyArrayIterObject *) PyArray_IterAllButAxis(py_sigma, &caxis);
else {
  sptr = new stype[1];
  sptr[0] = 1.0;
}
if(has_mask)
  mskitr = (PyArrayIterObject *) PyArray_IterAllButAxis(py_mask, &caxis);
  

ystride = Sy[caxis];
sstride = has_sigma ? Ssigma[caxis] : 0;
xstride = has_x ? Sx[caxis] : sizeof(xtype);
mstride = has_mask ? Smask[caxis] : sizeof(mtype);
unmasked = new int[ylen];


#define Yytype(pt) (*(yptr + (pt)*ystride/sizeof(ytype)))
// this could be arranged to always return 1.0
#define Sstype(pt) (*(sptr + (pt)*sstride/sizeof(stype)))
// this could be arranged to always return pt
#define Xxtype(pt) (*(xptr + (pt)*xstride/sizeof(xtype)))
#define MSKmtype(pt) (*(mskptr + (pt)*mstride/sizeof(mtype)))

while( PyArray_ITER_NOTDONE(yitr) ) {
  yptr = (ytype *) yitr->dataptr;
  mptr = (double *) PyArray_ITER_DATA(mitr);
  bptr = (double *) PyArray_ITER_DATA(bitr);
  chiptr = (double *) PyArray_ITER_DATA(chitr);
  if(has_x) xptr = (xtype *) PyArray_ITER_DATA(xitr);
  if(has_sigma) sptr = (stype *) PyArray_ITER_DATA(sitr);
  //else already exists
  
  // if 2 or more unmasked pts here, proceed
  // else b, m = 0 and chisq = whatever
  if(has_mask) {
    un_count = 0;
    mskptr = (mtype *) PyArray_ITER_DATA(mskitr);
    for(k=0; k<ylen; k++) {
      if(MSKmtype(k))
        unmasked[un_count++] = k;
    }
  } else {
    un_count = ylen;
    for(k=0; k<un_count; k++) unmasked[k] = k;
  }
  if(un_count < 2) {
    *mptr = 0.0;
    *bptr = 0.0;
    *chiptr = -1.0;
    PyArray_ITER_NEXT(yitr);
    PyArray_ITER_NEXT(mitr);
    PyArray_ITER_NEXT(bitr);
    PyArray_ITER_NEXT(chitr);
    if(has_x) PyArray_ITER_NEXT(xitr);
    if(has_sigma) PyArray_ITER_NEXT(sitr);
    if(has_mask) PyArray_ITER_NEXT(mskitr);
    continue;
  }
  sx = sy = s = sxx = sxy = 0.0;
  for(k=0; k<un_count; k++) {
    p = unmasked[k];
    y1 = Yytype(p);
    sig = Sstype(p);
    x1 = Xxtype(p);
    sx += x1/sig;
    sy += y1/sig;
    s += 1.0/sig;
    sxx += (x1 * x1)/sig;
    sxy += (x1 * y1)/sig;
  }
  delta = sxx * s - sx * sx;
  *bptr = (sxx * sy - sx * sxy)/delta;
  *mptr = (s * sxy - sx * sy)/delta;
  *chiptr = 0.0;
  for(k=0; k<un_count; k++) {
    p = unmasked[k];
    x1 = Xxtype(p);
    diff = Yytype(p) - (*mptr * x1 + *bptr);
    *chiptr += (diff * diff);
  }
  
  PyArray_ITER_NEXT(yitr);
  PyArray_ITER_NEXT(mitr);
  PyArray_ITER_NEXT(bitr);
  PyArray_ITER_NEXT(chitr);
  if(has_x) PyArray_ITER_NEXT(xitr);
  if(has_sigma) PyArray_ITER_NEXT(sitr);
  if(has_mask) PyArray_ITER_NEXT(mskitr);
}

delete [] unmasked;
if(!has_x) delete [] xptr;
if(!has_sigma) delete [] sptr;
"""
    mod = ext_tools.ext_module('util_ext')
    dtype2ctype = {
        np.dtype(np.float64): 'double',
        np.dtype(np.float32): 'float',
        }

    # m, b, and chisq are always float64
    m = np.empty((1,), np.float64)
    b = np.empty((1,), np.float64)
    chisq = np.empty((1,), np.float64)
    for dt in [np.dtype('d'), np.dtype('f')]:
        func_name = 'lin_regression_'+dt.char
        func_code = code.replace('ytype', dtype2ctype[dt])
        func_code = func_code.replace('xtype', dtype2ctype[dt])
        func_code = func_code.replace('mtype', dtype2ctype[dt])
        func_code = func_code.replace('stype', dtype2ctype[dt])
        locals_dict = {}
        locals_dict['y'] = np.empty((1,), dt)
        locals_dict['x'] = np.empty((1,), dt)
        locals_dict['mask'] = np.empty((1,), dt)
        locals_dict['sigma'] = np.empty((1,), dt)
        locals_dict.update(dict(has_x=1, has_sigma=1, has_mask=1,
                                m=m, b=b, chisq=chisq, caxis=0))
        # remember to preserve the function signature
        args = ['y', 'x', 'has_x', 'sigma', 'has_sigma',
                'mask', 'has_mask', 'm', 'b', 'chisq', 'caxis']
        modfunc = ext_tools.ext_function(func_name, func_code,
                                         args, local_dict=locals_dict)
        mod.add_function(modfunc)
    mod.customize.set_compiler('gcc')
    loc = split(__file__)[0]
    kw = {'include_dirs': [np.get_include()]}
    if build:
        mod.compile(location=loc, **kw)
    else:
        ext = mod.setup_extension(location=loc, **kw)
        ext.name = 'recon.' + ext.name
        return ext
#-----------------------------------------------------------------------------
@loads_extension_on_call('util_ext', locals())
def lin_regression(y, x=None, sigma=None, mask=None, axis=-1):
    """(m,b,chisq) = lin_regression(y, x=None, sigma=None, mask=None, axis=-1)
    
    Performs a linear regression on all lines of y along a given axis,
    returning slope, intercept, and chi squared error for each. EG, if
    y.shape = (12,40,100)
    m,b,chisq = lin_regression(y, axis=-1)
    m.shape == b.shape == chisq.shape == (12,40,1)

    Optional arrays are:
    x : the sampling points of y
    sigma : the sampling error of y (variance)
    mask : mask==0 at points to be thrown out of the analysis

    """
    yshape = list(y.shape)
    ndim = len(yshape)
    caxis = (axis + ndim) % ndim

    # enforce that all arrays have the same floating point type
    yt = y.dtype
    if x is None:
        x, has_x = (np.array([0], yt), 0)
    else:
        has_x = 1
    if sigma is None:
        sigma, has_sigma = (np.array([0], yt), 0)
    else:
        has_sigma = 1
    if mask is None:
        mask, has_mask = (np.array([0], yt), 0)
    else:
        has_mask = 1

    [x, sigma, mask] = [a.astype(yt) for a in [x, sigma, mask]]
    
    ax_len = yshape.pop(caxis)
    m = np.zeros(yshape, np.float64)
    b = np.zeros(yshape, np.float64)
    chisq = np.zeros(yshape, np.float64)

    func = eval('util_ext.' + 'lin_regression_' + yt.char)
    func(y, x, has_x, sigma, has_sigma, mask, has_mask, m, b, chisq, caxis)
    
    yshape.insert(caxis, 1)
    m.shape = yshape
    b.shape = yshape
    chisq.shape = yshape
    return (m, b, chisq)

#-----------------------------------------------------------------------------
def L1cost(x,y,m):
    b = np.median(y-m*x)
    sgn = np.sign(y - (m*x + b))
    return (x*sgn).sum(), b

def medfit(y, x=None):
    npt = y.shape[0]
    if npt < 2:
        print "impossible to solve"
        return (0.,0.,1e30)
    if x is None:
        x = np.arange(npt)
    (mm,b1,chisq) = lin_regression(y, x=x)
    mm = mm[0]
    b1 = b1[0]
    chisq = chisq[0]
    sigb = np.sqrt(chisq / (npt * np.power(x,2).sum() - x.sum()**2))
    m1 = mm
##     print "stdev:", sigb
    f1,b = L1cost(x,y,m1)
    m2 = m1 + np.sign(f1)*(3 * sigb)
    f2,_ = L1cost(x,y,m2)
##     print "initial bracket vals:", (f1,f2)
    while(f1*f2 > 0):
        mm = 2*m2 - m1
        m1 = m2
        f1 = f2
        m2 = mm
        f2,_ = L1cost(x,y,m2)
    sigb *= 0.01
    fstore = 0.
    reps = 0
    while abs(m2-m1) > sigb:
        mm = 0.5*(m1+m2)
        if mm==m1 or mm==m2:
            break
        f,b = L1cost(x,y,mm)
        if f==fstore:
            reps += 1
##             if reps > 2:
##                 print "breaking due to repeat f(b) hits"
##                 break
        else:
            fstore = f
            reps = 0
        if f*f1 > 0:
            f1 = f
            m1 = mm
        else:
            f2 = f
            m2 = mm
    m = mm
    absdev = np.abs(y - (m*x + b)).sum()
##     print "reps:", reps
    return m,b,absdev

#-----------------------------------------------------------------------------
def polyfit(x, y, deg, sigma=None, rcond=None, full=False):
    """Least squares polynomial fit.

    Yoinked from numpy and modified to minimize with
    variance information accounted for.

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
        sigma = np.ones(y.shape[-1])
    inv_stdev = np.power(sigma, -0.5)
    x = np.asarray(x) + 0.0
    y = (np.asarray(y) + 0.0)*inv_stdev

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
        if xtype == np.single or xtype == np.csingle :
            rcond = len(x)*np.finfo(np.single).eps
        else :
            rcond = len(x)*np.finfo(np.double).eps

    # scale x to improve condition number
    scale = abs(x).max()
    if scale != 0 :
        x /= scale

    # solve least squares equation for powers of x
    v = np.vander(x, order)*inv_stdev[:,None]
    c, resids, rank, s = np.linalg.lstsq(v, y, rcond)

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        import warnings
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    # scale returned coefficients
    if scale != 0 :
        c /= np.vander([scale], order)[0]

    if full :
        return c, resids, rank, s, rcond
    else :
        return c

#-----------------------------------------------------------------------------
def polyfit_2d(z, x, y, deg, mask=None, rcond=None):

    # a little helper function to break vector or number L into two indices
    def decompose(l,L):
        return np.floor(l/L).astype(np.int32), (l%L).astype(np.int32)

    M,N = z.shape
    if y.shape[0]!=M or x.shape[0]!=N:
        raise ValueError("dimensions of data vector don't match the grid dimensions")

    if mask is None:
        mask = np.ones(z.shape)
    unmasked = mask.reshape((M*N,)).nonzero()[0]

    ztype = z.dtype
    z.shape = (M*N,)
    powers = [zip(range(o,-1,-1), range(0,o+1)) for o in range(deg+1)]
    powers = reduce(lambda x,y: x+y, powers)
    ncol = len(powers)

    # set rcond
    if rcond is None :
        if ztype == np.single or ztype == np.csingle :
            #rcond = N*M*np.finfo(np.single).eps
            rcond = 10.*np.finfo(np.single).eps
        else :
            #rcond = N*M*np.finfo(np.double).eps
            rcond = 10.*np.finfo(np.double).eps


    yi,xi = decompose(np.arange(N*M), N)
    A = np.empty((M*N, ncol), dtype=ztype)

    # scale dim vectors so that x**(deg) isn't crazy big
    x = x.astype(ztype)
    y = y.astype(ztype)
    scale = (np.abs(x.max()), np.abs(y.max()))
    x /= scale[0]
    y /= scale[1]
    xax = x[xi]
    yax = y[yi]

    col_scale = np.power(scale[0], np.array([p[0] for p in powers])) * \
                np.power(scale[1], np.array([p[1] for p in powers]))

    # construct columns of A iteratively? x varies faster than y
    for c, p in enumerate(powers):
        A[:,c] = np.power(xax,p[0])*np.power(yax,p[1])

    Asub = A[unmasked]
    zsub = z[unmasked]
    c, resids, rank, s = np.linalg.lstsq(Asub, zsub, rcond)

    c /= col_scale
    A *= col_scale
    z.shape = (M,N)
    return A, c, resids, rank, s

def bivariate_fit(z, dim0, dim1, deg, sigma=None, mask=None,
                  rcond=None, scale=None):
    """Want to solve this problem:
    z[l,m] = sum_q,r (a[q,r]*dim0[l]^q*dim1[m]^r)
    
    If dim0 is L points and dim1 is M points, reorder data into an P point
    vector, where P = LM (p = l*L + m)
    
    also, reorder the system of equations into K = QR (k = q*Q + r) equations
    
    Now the set of equations is z[p] = sum_k (a[k]*Xk(x[p]))
    where a[k] -> a[q,r] and Xk[x[p]] = dim0[l]^q * dim1[m]^r
    
    So design matrix A will be structured:
    | X0[dim00,dim10] X1[dim00,dim10] X2[dim00,dim10] ... XQR[dim00,dim10] |
    | X0[dim00,dim11] X1[dim00,dim11] X2[dim00,dim11] ... XQR[dim00,dim11] |
    |  .               .               .                   .               |
    |  .               .               .                   .               | 
    | X0[dim00,dim1M] X1[dim00,dim1M] X2[dim00,dim1M] ... XQR[dim00,dim1M] |
    |  .               .               .                   .               |
    |  .               .               .                   .               |
    | X0[dim0L,dim1M] X1[dim0L,dim1M] X2[dim0L,dim1M] ... XQR[dim0L,dim1M] |
    
    """
    # a little helper function to break vector or number L into two indices
    def decompose(l,L):
        return np.floor(l/L).astype(np.int32), (l%L).astype(np.int32)
    
    dim0 = np.asarray(dim0) + 0.0
    dim1 = np.asarray(dim1) + 0.0
    z = np.asarray(z) + 0.0
    L,M = dim0.shape[0],dim1.shape[0]
    if np.product(z.shape) != L*M:
        raise ValueError("dimensions of data vector don't match the grid dimensions")
    Q = R = int(deg) + 1

    # reorder the data, variance, and mask arrays into L*M length vectors
    if sigma is None:
        sigma = np.ones((L,M))
    inv_stdev = np.power(np.reshape(sigma, (L*M,)), -0.5)

    if mask is None:
        mask = np.ones(z.shape)
    unmasked = (np.reshape(mask, (L*M,))).nonzero()[0]

    z = np.reshape(z, (L*M,))*inv_stdev
    l, m = decompose(np.arange(L*M), M)
    q, r = decompose(np.arange(Q*R), R)

    # set rcond
    if rcond is None :
        ztype = z.dtype
        if ztype == np.single or ztype == np.csingle :
            rcond = L*M*np.finfo(np.single).eps
        else :
            rcond = L*M*np.finfo(np.double).eps

    # design matrix
    if not scale:
        scale = abs(dim0).max(), abs(dim1).max()

    dim0 /= scale[0]
    dim1 /= scale[1]
    
    A = np.power(np.outer(dim0[l], np.ones(Q*R)), q) * \
        np.power(np.outer(dim1[m], np.ones(Q*R)), r)

    A_calc = np.take(A*inv_stdev[:,None], unmasked, axis=0)

    c, resids, rank, s = np.linalg.lstsq(A_calc, np.take(z, unmasked), rcond)

    vec_scale = np.power(scale[0], q)*np.power(scale[1],r)
    c /= vec_scale

    return A*vec_scale, c

#-----------------------------------------------------------------------------
def maskbyfit(M, sigma, tol, tol_growth, mask):

    # I could cast this problem into one column of fits, with
    # a meta-mask that hits off entries that have already reached
    # a stable point (so they don't keep getting recomputed)

    Mshape = M.shape
    nrows, npts = np.product(Mshape[:-1]), Mshape[-1]
    M = np.reshape(M, (nrows, npts))
    sigma = np.reshape(sigma, (nrows, npts))
    mask = np.reshape(mask, (nrows, npts))
    meta_mask = np.ones(nrows)

    mask_start, mask_end = np.zeros(nrows), np.ones(nrows)
    xax = np.outer(np.ones(nrows), np.arange(npts))
    n = 0
    while meta_mask.any():
        um = meta_mask.nonzero()[0]
        M_sub = M[um]
        xax_sub = xax[um]
        mask_sub = mask[um]        
        mask_start = mask.sum(axis=-1)
        if not mask_start.any():
            return
        # only compute where meta-mask is unmasked
        (m,b,chisq) = lin_regression(M_sub*mask_sub, X=xax_sub*mask_sub,
                                     sigma=sigma[um], mask=mask_sub)
        
        fitted = xax_sub * m[...,None] + b[...,None]
        np.putmask(mask_sub, abs(M_sub-fitted) > tol*chisq[...,None], 0)
        mask[um] = mask_sub
        mask_end = mask.sum(axis=-1)
        np.putmask(meta_mask, (mask_end==mask_start), 0)
        tol = tol*tol_growth
        n += 1
    M = np.reshape(M, Mshape)
    sigma = np.reshape(sigma, Mshape)
    mask = np.reshape(mask, Mshape)
    return

#-----------------------------------------------------------------------------
def unwrap_ref_volume(vol, fe1=None, fe2=None):
    """
    unwrap phases one "slice" at a time, where the volume
    is sliced along a single pe line (dimensions = nslice X n_fe)
    take care to move all the surfaces to roughly the same height
    @param phases is a volume of wrapped phases
    @return: uphases an unwrapped volume, shrunk to masked region
    """
    oslice = (fe1 is not None and fe2 is not None) and \
             (slice(None),slice(None),slice(fe1,fe2)) or \
             (slice(None),)*3
    uphases = np.empty(vol.shape, np.float64)
    zeropt = vol.shape[2]/2
    # search for best sl to be s=0 by looking at all s where q2=0,q1=0
    scut = abs(vol[:,vol.shape[1]/2,zeropt])
    zerosl = np.nonzero(scut == scut.max())[0][0]
    phases = np.angle(vol)
    # unwrap the volume sliced along each PE line
    # the middle of each surface should be between -pi and pi,
    # if not, put it there!
    for u in range(0,vol.shape[1],1):
        uphases[:,u,:] = unwrap2D(phases[:,u,:])
        height = uphases[zerosl,u,zeropt]
        height = int((height+np.sign(height)*np.pi)/2/np.pi)
        uphases[:,u,:] = uphases[:,u,:] - 2*np.pi*height

    return uphases[oslice]

#-----------------------------------------------------------------------------
def find_ramp(x, mask=None, do_unwrap=False, debug=False):
    """Fit a slope term to the (phase) data x. Since the data may exhibit
    patches of noise, which interrupts any attempt to unwrap phase jumps
    through segments, fit an adaptive number of intercept terms according
    to the number of low noise segments.
    """
    if mask is None:
        # THIS DEFAULT MASK NEEDS WORK
        d2x_sq = np.diff(x, n=2)**2
        min_sm = d2x_sq.min()
        mask = np.zeros(x.shape, 'i')
        mask[2:] = np.where(d2x_sq > np.var(x)/10., 0, 1)
        # trust this mask
        do_unwrap = True

    x_pts = mask.nonzero()[0]
    x_unmasked = x[x_pts]
    # The phase data may exhibit phase wraps inside of or between
    # various contiguous signal segments separated by noise patches.
    # In order to side-step the difficulty of unwrapping unknown
    # phase wraps between segments:
    # Construct a model system where the slope term remains constant
    # through many inter-contiguous segments, but the intercept term
    # may float between segments. 

    segs = [0]
    segs += [i+1 for i,n in enumerate(np.diff(x_pts)) if n>1]
    segs += [len(x_pts)]
    n_segs = len(segs)-1
    seg_pairs = [(segs[i],segs[i+1]) for i in xrange(n_segs)]
    obs_x = np.zeros(len(x_pts), 'd')
    A = np.zeros((len(x_pts), n_segs+1), 'd')
    A[:,0] = x_pts
    col = 1
    for pair in seg_pairs:
        start, stop = pair
        if do_unwrap:
            obs_x[start:stop] = np.unwrap(x_unmasked[start:stop])
        else:
            obs_x[start:stop] = x_unmasked[start:stop]
        A[start:stop,col] = 1
        col += 1

    # solve for [slope + n_seg intercepts]
    r = np.linalg.lstsq(A, obs_x)[0]
    m = r[0]
    if debug:
        import matplotlib.pyplot as pp
        pp.plot(np.arange(len(x)), x)
        pp.plot(x_pts, x_unmasked, 'r.')
        pp.plot(x_pts, np.dot(A, r), 'g-')
        pp.show()
    return m
    

#-----------------------------------------------------------------------------
#scipy's unwrap (pythonication of Matlab's routine)
def unwrap1D(p,discont=np.pi,axis=-1,return_diffs=False):
    """unwraps radian phase p by changing absolute jumps greater than
       discont to their 2*pi complement along the given axis.
    """
    p = np.asarray(p)
    nd = len(p.shape)
    dd = np.diff(p,axis=axis)
    slice1 = [slice(None,None)]*nd     # full slices
    slice1[axis] = slice(1,None)
    ddmod = np.mod(dd+(2*np.pi-discont),2*np.pi)-(2*np.pi-discont)
    np.putmask(ddmod,(ddmod==-np.pi) & (dd > 0),np.pi)
    ph_correct = ddmod - dd
    np.putmask(ph_correct,abs(dd)<discont,0)
    up = np.array(p,copy=1, dtype=np.float64)
    up[slice1] = p[slice1] + np.add.accumulate(ph_correct,axis)
    if not return_diffs:
        return up
    else:
        return np.add.accumulate(ph_correct,axis)

#-----------------------------------------------------------------------------
def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using FFT. See convolve.
    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
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
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return _centered(ret,osize,axes=axes)
    elif mode == "valid":
        return _centered(ret,abs(s2-s1)+1,axes=axes)

#-----------------------------------------------------------------------------
def _centered(arr, newsize, axes=None):
    # Return the center newsize portion of the array.
    ndim = len(newsize)
    if axes is None:
        axes = range(ndim)
    newsize = np.asarray([newsize[a] for a in axes])
    currsize = np.array([arr.shape[a] for a in axes])
    startind = (currsize - newsize) / 2 # + 1
    endind = startind + newsize
    myslice = [slice(None)]*(ndim-len(axes))
    myslice += [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

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
            iy = np.clip(np.floor(y).astype(np.int32), 0, n_pe-2)
            dy = y - iy
            for m in range(n_pe):
                s[:,m]=((1.-dy[:,m])*np.take(s[:,m],iy[:,m]) \
                        + dy[:,m]*np.take(s[:,m],iy[:,m]+1)).astype(np.float32)

    return vol_data

#-----------------------------------------------------------------------------
#---------- LINEAR SOLUTION REGULARIZATION ROUTINES --------------------------
def regularized_solve(A, b, l):
    """Solves Ax=b for x using simplified Tikhonov regularization with
    parameter lambda = l. The solution is:
    x = (A' + l**2*I)^-1 * (A'b), where A' is the Hermitian transpose of A
    """
    cplx = A.dtype.char in ['F', 'D']
    if cplx:
        A2 = np.dot(A.conjugate().T, A)
    else:
        A2 = np.dot(A.T, A)
    n = A2.shape[0]
    l = l*l
    A2.flat[0:n*n:n+1] += l
    if cplx:
        b2 = np.dot(A.conjugate().T, b)
    else:
        b2 = np.dot(A.T, b)
    return np.linalg.solve(A2,b2)

    

def lcurve(A, b, lm_range=None):
    """returns error functions for two minimization problems:
    |Ax - b|**2 and |x|**2
    where the functions are parameterized by reg. factor lambda
    """
    [u,s,vt] = np.linalg.svd(A, 0, 1)
    if lm_range is None:
        lm_range = np.power(10., np.linspace(-10, 1, 100))
    fi = s[None,:]**2 / (s[None,:]**2 + lm_range[:,None]**2)
    eta = np.zeros(len(lm_range))
    rho = np.zeros(len(lm_range))
    d_eta = np.zeros(len(lm_range))
    for i in xrange(len(lm_range)):
        xl = np.dot((fi[i]/s)[:,None]*u.conjugate().transpose(), b)
        eta[i] = np.dot(xl, xl.conjugate()).real
        xp = 0.
        for n in xrange(len(s)):
            d = np.dot(u[:,n].conjugate().transpose(), b)
            d = (d*d.conjugate()).real
            xp += d*(1-fi[i,n])*(fi[i,n]**2)/(s[n]**2)
        d_eta[i] = -xp*4/lm_range[i]
        br = np.dot((1-fi[i])[:,None]*u.transpose().conjugate(), b)
        rho[i] = np.dot(br, br.conjugate()).real

    lm2 = lm_range**2
    lm4 = lm_range**4
    eta2 = eta**2
    rho2 = rho**2
    kappa = 2*eta*rho*(lm2*d_eta*rho + 2*lm_range*eta*rho + lm4*eta*d_eta)
    kappa /= d_eta * ((lm2*eta2 + rho2)**1.5)
    return eta, rho, kappa

def lcurve2(A, b, lm_range=None, L=None, u=None, s=None):
    """returns error functions for two minimization problems:
    |Ax - b|**2 and |x|**2
    where the functions are parameterized by reg. factor lambda
    """
    from scipy.linalg import flapack, gsvd
    if u is None or s is None:
        if L is None:
            u,s,_ = np.linalg.svd(A, 0, 1)
        else:
            L = L.astype(A.dtype)
            p,n = L.shape
            [u,v,q,c,s,nr] = gsvd(A,L)
            alpha = c.real.diagonal()[n-p:]
            beta = s[:,n-p:].real.diagonal()
##             k,l,u,v,q,alpha,beta,iwork,info = flapack.zggsvd(A, L)
            s = np.array([alpha, beta])
    if lm_range is None:
        lm_range = np.power(10., np.linspace(-10, 1, 100))
    eta = np.zeros(len(lm_range))
    rho = np.zeros(len(lm_range))
    kappa = np.zeros(len(lm_range))
    for i in xrange(len(lm_range)):
        eta[i], rho[i], kappa[i] = eta_rho_kappa(lm_range[i], u, s, b)
    return eta, rho, kappa
        
def eta_rho_kappa(l, u, sm, b):
    l = np.array(l, dtype=sm.dtype).reshape(())
    beta = np.dot(u.transpose().conjugate(), b)
    if len(sm.shape) > 1:
        s = sm[0]/sm[1]
        beta = beta[b.shape[0]-s.shape[0]:]
    else:
        s = sm
    f = s**2 / (s**2 + l**2)
    #f = s/(s+l)
    cf = 1-f
    beta2 = np.dot(b.flat[:],b.flat[:].conjugate()).real - \
            np.dot(beta.flat[:],beta.flat[:].conjugate()).real
    if len(b.shape) > 1:
        xi = beta/s[:,None]
        e = f[:,None]*xi
        br = cf[:,None]*beta
    else:
        xi = beta/s
        e = f*xi
        br = cf*beta

    eta = np.dot(e.flat[:], e.flat[:].conjugate()).real ** 0.5
    rho = np.dot(br.flat[:], br.flat[:].conjugate()).real ** 0.5
    if u.shape[0] > u.shape[1] and beta2:
        rho = (rho**2 + beta2)**0.5
##     xl = np.dot((f/s)[:,None]*u.transpose().conjugate(), b)
##     eta = np.dot(xl.flat[:], xl.flat[:].conjugate()).real
##     br = np.dot((1-f)[:,None]*u.transpose().conjugate(), b)
##     rho = np.dot(br.flat[:], br.flat[:].conjugate()).real

    if len(b.shape) > 1:
        # transpose for simpler broadcasting
        xi = xi.T
        beta = beta.T
    f1 = -2*f*cf/l
    f2 = -f1*(3-4*f)/l
    axi2 = np.abs(xi)**2
    abeta2 = np.abs(beta)**2
    phi = (f*f1*axi2).sum()
    psi = (cf*f1*abeta2).sum()
    dphi = ((f1**2 + f*f2)*axi2).sum()
    dpsi = ((-f1**2 + cf*f2)*abeta2).sum()
    
    d_eta = phi/eta
    d_rho = -psi/rho
    dd_eta = dphi/eta - d_eta*d_eta/eta
    dd_rho = -dpsi/rho - d_rho*d_rho/rho
    dlogeta = d_eta/eta; dlogrho = d_rho/rho
    ddlogeta = dd_eta/eta - dlogeta**2
    ddlogrho = dd_rho/rho - dlogrho**2
    kappa = (dlogrho*ddlogeta - ddlogrho*dlogeta)/(dlogrho**2 + dlogeta**2)**(1.5)
##     d_eta = np.array(0., dtype=s.dtype)
##     for n in xrange(len(s)):
##         d = np.dot(u[:,n].transpose().conjugate(), b)
##         if d.shape:
##             d = np.dot(d.flat[:], d.flat[:].conjugate()).real
##         else:
##             d = (d*d.conjugate()).real
##         d_eta += d*(1-f[n])*(f[n]**2)/(s[n]**2)
##     d_eta *= -4/l

##     lm2 = l**2
##     lm4 = l**4
##     eta2 = eta**2
##     rho2 = rho**2
##     kappa = 2*eta*rho*(lm2*d_eta*rho + 2*l*eta*rho + lm4*eta*d_eta)
##     kappa /= d_eta * ( (lm2*eta2 + rho2)**(1.5) )
    return eta, rho, kappa

def eta(*args):
    return eta_rho_kappa(*args)[0]

def rho(*args):
    return eta_rho_kappa(*args)[1]

def kappa(*args):
    return eta_rho_kappa(*args)[2]

def kappa_opt(*args):
    # it's helpful to optimize over the exponential range..
    l = 10**args[0]
    fargs = (l,) + args[1:]
    return -kappa(*fargs)

def regularized_solve_lcurve(A, b, L=None, u=None, s=None, vt=None,
                             max_lx_norm=None):
    """ Find a solution x that minimizes the expression
    ||A*x - b||{2} + lm**2||Lx||{2}
    
    if L is None, then take L as eye(n):
      u, s, vt = svd(A)
    else
      u1, u2, x, s1, s2 = gsvd_matlab(A, L)
      p,n = L.shape
      u <-- u1
      s <-- array([diag(s1.real), diag(s2[:,n-p:].real)])
      vt <-- x

    setting max_lx_norm is helpful to constrain the range of the
    solution fit versus solution norm curvature function.
    """
    import scipy as sp
    from scipy.linalg import gsvd_matlab
    cplx = A.dtype.char in ['F', 'D']
    
##     if cplx:
##         [u,s,vt] = np.linalg.svd(A.conjugate(), 0, 1)
##     else:
    if u is None or s is None or vt is None:
        if L is None:
            [u,s,vt] = np.linalg.svd(A, 0, 1)
        else:
            L = L.astype(A.dtype)
            p,n = L.shape
            [u,u2,vt,c,s] = gsvd_matlab(A,L)
            alpha = c.real.diagonal()[n-p:]
            beta = s[:,n-p:].real.diagonal()
##             k,l,u,v,q,alpha,beta,iwork,info = flapack.zggsvd(A, L)
            s = np.array([alpha, beta])
            

    if max_lx_norm is not None:
        # it seems
        # necessary to get a better estimate for the global min of -kappa.
        # It also might be a better idea to set the lambda range a la Hansen
        n = -10
        while eta(10.0 ** n, u, s, b) > max_lx_norm:
            n += 1
        
        lm_lims = (n-1, n+4)
        
##         lm_range = np.power(10., np.linspace(-10, 1, 100))
##         eta, rho, kappa = lcurve2(A, b, lm_range=lm_range, u=u, s=s)
##         bn = np.dot(b.flat[:],b.flat[:].conj()).real**0.5
##         good_idx = np.argwhere(eta <= 10*bn).flatten()
##         n = np.argmax(kappa[good_idx]) + good_idx[0]
##         lm_lims = (np.log10(lm_range[max(0,n-5)]),
##                    np.log10(lm_range[min(99, n+5)]))
    else:
        lm_lims = (-10, 3)
    print lm_lims
    
    lm_exp = sp.optimize.fminbound(kappa_opt, lm_lims[0], lm_lims[1],
                                   args=(u,s,b))
    lm_opt = 10**lm_exp
    print 'lm_opt:', lm_opt
    if L is None:
        ireg_svals = s / (s**2 + lm_opt**2)
        xreg = np.dot(vt.conjugate().T,
                      np.dot(ireg_svals[:,None]*u.conjugate().T, b))
    else:
        p,n = L.shape
        vi = np.linalg.inv(vt.conj().T)
        bb = np.dot(u[:,n-p:].conj().T, b)
        bb *= s[0]
        gm2 = (s[0]/s[1])**2
        if n==p:
            x0 = np.zeros((n,), b.dtype)
        else:
            x0 = np.dot(vi[:,:n-p], np.dot(u[:,:n-p].conj().T, b))
        xi = bb/(s[0]**2 + lm_opt**2 * s[1]**2)
        xreg = np.dot(vi[:,n-p:], xi) + x0
    
    return xreg, lm_opt
    
def regularized_inverse(A, lm):
    """
    This computes the gereralized inverse of A through a 
    regularized solution: (Ah*A + lm^2*I)*C = Ah*I

    The solution is computed destructively (A becomes (Ainv)^h)
    
    use CBLAS to get the matrix product of (Ah*A + lm^2*I) in A2,
    and then use LAPACK to solve for C. 
    
    Note: in column-major...
    A is conj(Ah) 
    A2 is conj(Ah*A + lm^2*I) -- since A2 is hermitian symmetric
    
    If conj(AhA + (lm^2)I)*C = conj(Ah), then conj(C) (the conjugate of
    the LAPACK solution, which is in col-major) is the desired solution
    in column-major. So the final answer in row-major is the hermitian
    transpose of C.
    """
    from numpy.linalg import lapack_lite
    m,n = A.shape
    cplx = A.dtype.char in ['F', 'D']
    # Ah is NxM 
    Ah = np.empty((n,m), A.dtype)
    if cplx:
        Ah[:] = A.conjugate().T
    else:
        Ah[:] = A.T

    # A2 is NxN
    A2 = np.dot(Ah,A)
    # add lm**2 to the diagonal (ie, A2 + (lm**2)*I)
    A2.flat[0:n*n:n+1] += (lm*lm)

    pivots = np.zeros(n, np.intc)

    # A viewed as column major is considered to be NxNRHS (NxM)...
    # the solution will be NxNRHS too, and is stored in A
    solver = lapack_lite.zgesv and cplx or lapack_lite.dgesv
    results = solver(n, m, A2, n, pivots, A, n, 0)
    
    # put conjugate solution into row-major (MxN) by hermitian transpose
    if cplx:
        Ah[:] = A.conjugate().T
    else:
        Ah[:] = A.T
    return Ah

def regularized_inverse_smooth(A, lm):
    from numpy.linalg import lapack_lite
    m,n = A.shape
    cplx = A.dtype.char in ['F', 'D']
    Ah = np.empty((n,m), A.dtype)
    if cplx:
        Ah[:] = A.conjugate().T
    else:
        Ah[:] = A.T

    # B is a forward differencing operator
    B = np.zeros((n-1,n))
    B.flat[0:(n-1)*n:n+1] = -1.
    B.flat[1:(n-1)*n:n+1] =  1.

    A2 = np.dot(Ah,A)
    B = (lm*lm)*np.dot(B.T,B)
    A2 += B

    pivots = np.zeros(n, np.intc)

    solver = lapack_lite.zgesv and cplx or lapack_lite.dgesv    
    results = solver(n, m, A2, n, pivots, A, n, 0)

    if cplx:
        Ah[:] = A.conjugate().T
    else:
        Ah[:] = A.T
    return Ah
    
#-----------------------------------------------------------------------------
# quaternion and euler rotation helpers
class Quaternion:
    def __init__(self, i=0., j=0., k=0., qfac=1., M=None):
        self.Q = None
        if M is not None:
            self.matrix2quat(np.asarray(M))
        else:
            self.Q = np.array([i, j, k])
            self.qfac = qfac

    def matrix2quat(self, m):
        # m should be 3x3
        if len(m.shape) != 2:
            raise ValueError("Matrix must be 3x3")

        M = m[-3:,-3:].astype('d')

        ru2x, rv2x, rw2x, ru2y, rv2y, rw2y, ru2z, rv2z, rw2z = M.flat
        if np.linalg.det(M) < 0:
            rw2x *= -1; rw2y *= -1; rw2z *= -1
            self.qfac = -1
        else:
            self.qfac = 1

        K = np.array([ [ru2x-rv2y-rw2z, rv2x+ru2y, rw2x+ru2z, rv2z-rw2y],
                       [rv2x+ru2y, rv2y-ru2x-rw2z, rw2y+rv2z, rw2x-ru2z],
                       [rw2x+ru2z, rw2y+rv2z, rw2z-ru2x-rv2y, ru2y-rv2x],
                       [rv2z-rw2y, rw2x-ru2z, ru2y-rv2x, ru2x+rv2y+rw2z] ])/3.
        vals, vecs = np.linalg.eig(K)
        
        # Select largest eigenvector, reorder
        q = vecs[[3, 0, 1, 2],np.argmax(vals)]
        if q[0] < 0:
            q *= -1
        self.Q = q[1:]
##         print q
##         self.qfac = 1
        
##         xd = np.sqrt(M[0,0]**2 + M[1,0]**2 + M[2,0]**2)
##         yd = np.sqrt(M[0,1]**2 + M[1,1]**2 + M[2,1]**2)
##         zd = np.sqrt(M[0,2]**2 + M[1,2]**2 + M[2,2]**2)
##         if xd < 1e-5:
##             M[:,0] = np.array([1., 0., 0.])
##             xd = 1.
##         if yd < 1e-5:
##             M[:,1] = np.array([0., 1., 0.])
##             yd = 1.
##         if zd < 1e-5:
##             M[:,2] = np.array([0., 0., 1.])
##             zd = 1.
##         M[:,0] /= xd
##         M[:,1] /= yd
##         M[:,2] /= zd

##         if (np.dot(np.transpose(M), M) - np.identity(3)).sum() > 1e-5:
##             raise ValueError("matrix not orthogonal, must fix")

##         zd = np.linalg.det(M)
##         if zd > 0:
##             self.qfac = 1.0
##         else:
##             self.qfac = -1.0
##             M[:,2] *= -1.0

##         a = np.trace(M) + 1.0
##         if a > 0.5:
##             a = 0.5 * np.sqrt(a)
##             b = 0.25 * (M[2,1]-M[1,2])/a
##             c = 0.25 * (M[0,2]-M[2,0])/a
##             d = 0.25 * (M[1,0]-M[0,1])/a
##         else:
##             xd = 1.0 + M[0,0] - (M[1,1] + M[2,2])
##             yd = 1.0 + M[1,1] - (M[0,0] + M[2,2])
##             zd = 1.0 + M[2,2] - (M[0,0] + M[1,1])
##             if xd > 1.0:
##                 b = 0.5 * np.sqrt(xd)
##                 c = 0.25 * (M[0,1] + M[1,0])/b
##                 d = 0.25 * (M[0,2] + M[2,0])/b
##                 a = 0.25 * (M[2,1] - M[1,2])/b
##             elif yd > 1.0:
##                 c = 0.5 * np.sqrt(yd)
##                 b = 0.25 * (M[0,1] + M[1,0])/c
##                 d = 0.25 * (M[1,2] + M[2,1])/c
##                 a = 0.25 * (M[0,2] - M[2,0])/c
##             else:
##                 d = 0.5 * np.sqrt(zd)
##                 b = 0.25 * (M[0,2] + M[2,0])/d
##                 c = 0.25 * (M[1,2] + M[2,1])/d
##                 a = 0.25 * (M[1,0] - M[0,1])/d
##             if a < 0.01:
##                 (a, b, c, d) = (-a, -b, -c, -d)

##         self.Q = np.array([b, c, d])
        

    def tomatrix(self):
        Nq = np.dot(self.Q, self.Q)
        a = 1 - Nq
        # this is probably round off error.. 
        if 1 < Nq:
            a = (1/Nq)**.5
            if (a - np.floor(a)) < 0:
                a = np.floor(a)
            self.Q *= a
            a = 0
        else:
            a = (1 - Nq)**.5
        qa = np.array([a] + list(self.Q))
        Nq = np.dot(qa, qa)
        if Nq > 0.0:
            s = 2/Nq
        else:
            s = 0.0
        w, x, y, z = qa
        X = x*s
        Y = y*s
        Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
##         # transposed
##         m = np.array([[1.0-(yY+zZ), xY+wZ, xZ-wY],
##                       [xY-wZ, 1.0-(xX+zZ), yZ+wX],
##                       [xZ+wY, yZ-wX, 1.0-(xX+yY)]])
        m = np.array([[1.0-(yY+zZ), xY-wZ, xZ+wY],
                      [xY+wZ, 1.0-(xX+zZ), yZ-wX],
                      [xZ-wY, yZ+wX, 1.0-(xX+yY)]])
        if self.qfac < 0:
            m[:,2] *= -1
        return m

##         R = np.empty((3,3), np.float64)
##         b,c,d = tuple(self.Q)
##         a = 1.0 - (b*b + c*c + d*d)
##         if (a < 1.e-5):
##             print 'fixing a'
##             a = 1.0 / np.power(b*b+c*c+d*d, 0.5)
##             if (a - np.floor(a)) < 1.e-5:
##                 a = np.floor(a)
##             b *= a
##             c *= a
##             d *= a
##             a = 0. 
##         else:
##             a = np.power(a,0.5)
##         R[0,0] = (a*a + b*b - c*c - d*d)
##         R[0,1] = 2.*(b*c - a*d)
##         R[0,2] = 2.*(b*d + a*c)*self.qfac
##         R[1,0] = 2.*(b*c + a*d)
##         R[1,1] = (a*a + c*c - b*b - d*d)
##         R[1,2] = 2.*(c*d - a*b)*self.qfac
##         R[2,0] = 2.*(b*d - a*c)
##         R[2,1] = 2.*(c*d + a*b)
##         R[2,2] = (a*a + d*d - c*c - b*b)*self.qfac
##         R = np.where(abs(R - np.round(R)) < 1e-5, np.round(R), R)
##         return R
    
    def mult(self, quat):
        return Quaternion(M=np.dot(self.tomatrix(), quat.tomatrix()))

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
def real_euler_rot(phi=0, theta=0, psi=0):
    D = np.zeros((3,3), dtype='d')
    D[0,0] = D[1,1] = np.cos(phi)
    D[0,1] = np.sin(phi)
    D[1,0] = -np.sin(phi)
    D[2,2] = 1

    C = np.zeros((3,3), dtype='d')
    C[1,1] = C[2,2] = np.cos(theta)
    C[1,2] = np.sin(theta)
    C[2,0] = -np.sin(theta)
    C[0,0] = 1

    B = np.zeros((3,3), dtype='d')
    B[0,0] = B[1,1] = np.cos(psi)
    B[0,1] = np.sin(psi)
    B[1,0] = -np.sin(psi)
    B[2,2] = 1

    M = np.dot(B, np.dot(C, D))
    np.putmask(M, np.abs(M) < 1e-5, 0)
    return M

# should rename this.. 
def eulerRot(theta=0, psi=0, phi=0):
    """This returns a 3D rotation transform composed in the following way:
    NIFTI defines A = B*C*D = Ri(theta)*Rj(psi)*Rk(phi)
    so the quaternions will have this convention
    http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage
    """
    aboutI = np.zeros((3,3),np.float64)
    aboutJ = np.zeros((3,3),np.float64)
    aboutK = np.zeros((3,3),np.float64)
    # bank (clockwise, looking down axis)
    aboutI[0,0] = 1
    aboutI[1,1] = aboutI[2,2] = np.cos(theta)
    aboutI[1,2] = np.sin(theta)
    aboutI[2,1] = -np.sin(theta)
    # attitude (clockwise, looking down axis)
    aboutJ[1,1] = 1
    aboutJ[0,0] = aboutJ[2,2] = np.cos(psi)
    aboutJ[0,2] = -np.sin(psi)
    aboutJ[2,0] = np.sin(psi)
    # heading (clockwise, looking down axis -- x points left)
    aboutK[2,2] = 1
    aboutK[0,0] = aboutK[1,1] = np.cos(phi)
    aboutK[0,1] = np.sin(phi)
    aboutK[1,0] = -np.sin(phi)
    M = np.dot(aboutI, np.dot(aboutJ, aboutK))
    # make sure no rounding error proprogates from here
    np.putmask(M, abs(M)<1e-5, 0)
    return M

def decompose_rot(M):
    psi = -np.arcsin(M[0,2])
    if (abs(psi)-np.pi/2)**2 < 1e-9:
        theta = 0
        phi = np.arctan2(-M[1,0], -M[2,0]/M[0,2])
    else:
        c = np.cos(psi)
        theta = np.arctan2(M[1,2]/c, M[2,2]/c)
        phi = np.arctan2(M[0,1]/c, M[0,0]/c)
    return (theta, psi, phi)

def test_decompose_rot(M):
    (th, ps, ph) = decompose_rot(M)
    M2 = eulerRot(theta=th, psi=ps, phi=ph)
    assert np.allclose(M,M2), "not close: "+str(M)


#-----------------------------------------------------------------------------
def feval_bilinear(fx, t, dt, axis=0):
    t = np.asarray(t)
    if not t.shape:
        t.shape = (1,)
    npts = fx.shape[axis]
    lo = (t/dt).astype('i')
    hi = lo+1
    np.putmask(lo, lo<0, 0)
    np.putmask(hi, hi<0, 0)
    np.putmask(lo, lo>=npts, npts-1)
    np.putmask(hi, hi>=npts, npts-1)
    chi = t/dt - lo
    clo = hi - t/dt

    reslice_hi = [slice(None)] * len(fx.shape)
    reslice_lo = [slice(None)] * len(fx.shape)
    chi_slice = [None]*len(fx.shape)
    clo_slice = [None]*len(fx.shape)
    reslice_hi[axis] = hi
    reslice_lo[axis] = lo
    chi_slice[axis] = slice(None)
    clo_slice[axis] = slice(None)
    
    fi = (chi[chi_slice]*fx[reslice_hi] + clo[clo_slice]*fx[reslice_lo])
    return fi

class Gradient:
    TIMESTEP = 5.0 # microsec
    GAMMA = 2*np.pi*4258e4 # rad/s/Tesla
    def __init__(self, Tr, Tf, T0, n_echo, N1, Lx):
        self.Tr = Tr; self.Tf = Tf; self.T0 = T0
        self.gx = self.create_grad(Tr, Tf, n_echo)
        self.kx = self.integrate_grad()
        self.kx -= (Tr + Tf)/2.
        As = (Tf + Tr - T0**2/Tr) # microsec
        self.gmaG0 = 2*np.pi*N1/(As*Lx) # rad/microsec/mm
        self.dx = Lx/N1

    def create_grad(self, Tr, Tf, n_echo):
        npts_r = Tr/self.TIMESTEP
        npts_f = Tf/self.TIMESTEP
        lobe_pts = 2*npts_r + npts_f
        if npts_r:
            ramp_rate = 1.0/npts_r
        glen = lobe_pts*n_echo
        gx = np.zeros(glen)
        polarity = 1.0
        for lobe in range(n_echo):
            i0 = lobe*lobe_pts
            gx[i0:i0+npts_r] = polarity*ramp_rate*np.arange(npts_r)
            i0 += npts_r
            gx[i0:i0+npts_f] = polarity
            i0 += npts_f
            gx[i0:i0+npts_r] = polarity*ramp_rate*(lobe_pts - \
                                                   np.arange(npts_f+npts_r,
                                                             lobe_pts))
            polarity *= -1
        return gx

    def integrate_grad(self):
        kx = np.zeros_like(self.gx)
        for i in range(1,kx.shape[0]):
            kx[i] = self.TIMESTEP*(self.gx[i] + self.gx[i-1])/2.0 + kx[i-1]
        return kx

    def gxt(self, t):
        return feval_bilinear(self.gx, t, self.TIMESTEP)

    def kxt(self, t):
        return feval_bilinear(self.kx, t, self.TIMESTEP)

    def kxt_analytical(self, t):
        k0 = -(self.Tr+self.Tf)/2.
        r1 = t<self.Tr
        r2 = (t>=self.Tr) & (t<(self.Tr+self.Tf))
        r3 = t>=(self.Tf+self.Tr)
        kx = np.zeros_like(t)
        kx[r1] = t[r1]**2/(2*self.Tr)
        kx[r2] = self.Tr/2. - self.Tr + t[r2]
        kx[r3] = (self.Tf+self.Tr)-(2*self.Tr+self.Tf - t[r3])**2/(2*self.Tr)
        return kx + k0

    def gxt_analytical(self, t):
        r1 = t<self.Tr
        r2 = (t>=self.Tr) & (t<(self.Tr+self.Tf))
        r3 = t>=(self.Tf+self.Tr)
        gx = np.zeros_like(t)
        gx[r1] = t[r1]/self.Tr
        gx[r2] = 1.
        gx[r3] = (2*self.Tr+self.Tf-t[r3])/self.Tr
        return gx

    def gxt_proper(self, t):
        """Returns gx(t) in terms of milliTesla/cm
        """
        g = self.gxt(t)
        # rad/(us*mm) * (1e6us/s) * (1e1mm/cm) * (s*T/rad) * (1e3mT/T)
        G0 = self.gmaG0*1e10/self.GAMMA
        return G0*g

    def kxt_proper(self, t):
        """Returns kx(t) in terms of 1/cm
        """
        k = self.kxt(t)
        #  int<0,t>{g(t)*GAMMA/2PI} --->
        # Hz/T * mT/cm * us * (1e-3T/mT) * (1e-6us/s) ---> 1/cm
        
        # self.gmaG0*1e10/self.GAMMA * (self.GAMMA)/(2*np.pi) * 1e-9 -->
        kscl = self.gmaG0*10/(2*np.pi)
        return kscl * k


def grad_from_epi(epi):
    for timing_info in ['T_ramp', 'T_flat', 'T0']:
        assert hasattr(epi, timing_info)
    if epi.N1 == epi.n_pe and epi.fov_x > epi.fov_y:
        epi.fov_y *= 2
        epi.jsize = epi.isize
    elif epi.fov_y == epi.fov_x and epi.n_pe > epi.N1:
        epi.fov_y *= (epi.n_pe/epi.N1)
        epi.jsize = epi.isize
    print epi.fov_y
    return Gradient(epi.T_ramp, epi.T_flat, epi.T0,
                    epi.n_pe, epi.N1, epi.fov_x)
    

#-----------------------------------------------------------------------------
# MemmapArray class with automatic file deletion (usually!)

#def split_large_file(filename, dtype, offset=0)

class TempMemmapArray(np.memmap):
    """This is a light wrapper of Numpy's memmap, intended for producing an
    on-disk buffer. In addition to ndarray functionality, this memmap holds
    onto the file name created by the tempfile module so that it can be deleted
    when the memmap sees __del__.
    """
    def __new__(subtype, shape, dtype):
        f = tempfile.NamedTemporaryFile(prefix='rtools')
        data = np.memmap.__new__(subtype, f.name,
                                dtype=dtype, shape=shape, mode='r+')

        data = data.view(subtype)
        data.f = f
        return data

    def __array_finalize__(self, obj):
        np.memmap.__array_finalize__(self, obj)
        if hasattr(obj, 'f'):
            self.f = getattr(obj, 'f')
        if hasattr(obj, '_mmap'):
            self._mmap = obj._mmap

    def __del__(self):
        if type(self.base) is np.memmap:
            try:
                self.base._mmap.__del__()
                self.f.close()
                print 'closed tempfile', self.f.file.closed
            except ValueError, AttributeError:
                print "almost unlinked file in use"

## class PagedMemmapArray(np.ndarray):
##     """This is a simple wrapper of Numpy's memmap type of ndarray.

##     It overcomes the 2 GB limit on memory mapped files by one of two
##     paging strategies:

##     1) a big array "in memory" is mapped over several files on disk.
##     2) a big file on disk is mapped with several offsets.

##     The resulting layer is identically a set of memory pages that share a
##     common full_array_index --> page_array_index conversion.

##     Accesses that cross pages will have to be handled carefully!

##     nb: 'q' is the dtype character for int64
##     """
##     def __new__(subtype, shape, dtype, filename='', offset=0, mode='r'):
##         max_size = np.int64(2**31)
##         n_items = np.int64(shape[0])
##         for dim_size in shape[1:]:
##             n_items *= dim_size
##         dtype = np.dtype(dtype)

##         # Warning: pervasive integer division
##         page_items = max_size / dtype.itemsize
##         page_size = page_items * dtype.itemsize

##         n_pages = n_items / page_items
##         if n_pages * page_items < n_items:
##             n_pages += 1
##         # last page size might not be full size, so subtract off the overshoot
##         last_page_items = page_items - (n_pages*page_items - n_items)

##         data = np.empty((n_pages,), dtype=np.object_)

##         # now fix page shapes (best to flatten it and deal with access later?)
##         page_shape = (int(page_items),)
        
        
##         if filename:
##             # open a number of memmaps at different offsets
##             arr_offsets = np.arange(n_pages, dtype='q') * page_size + offset
##             for n in xrange(n_pages-1):
##                 data[n] = np.memmap(filename, dtype=dtype,
##                                     shape=page_shape, mode=mode,
##                                     offset = arr_offsets[n])
##                 #data[n] = data[n].view(subtype)
##             data[n_pages-1] = np.memmap(filename, dtype=dtype,
##                                         shape=(int(last_page_items),),
##                                         mode=mode,
##                                         offset = arr_offsets[n-1])
            
##         else:
##             # open a number of TempMemmapArrays
##             raise NotImplementedError

##         data = data.view(subtype)
##         data.n_pages = n_pages
##         data.arr_offsets = arr_offsets
##         return data

##     def __array_finalize(self, obj):
##         np.ndarray.__array_finalize__(self, obj)
##         if hasattr(obj, 'n_pages'):
##             self.n_pages = getattr(obj, 'n_pages')
##         if hasattr(obj, 'arr_offsets'):
##             self.arr_offsets = getattr(obj, 'arr_offsets')
        
    
        

        
#-----------------------------------------------------------------------------

if __name__ == "__main__":
    import doctest
    doctest.testmod()

