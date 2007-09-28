try:
    from _punwrap2D import Unwrap2D
    from _punwrap3D import Unwrap3D
except ImportError:
    raise ImportError("Please compile the C extensions to use this module")

import numpy as N

def unwrap2D(matrix, mask=None):
    """The method for this module unwraps a 2D grid of wrapped phases
    using the quality-map unwrapper.
    @param matrix, if ndim > 2, explode; if ndim < 2, a 1xN matrix
    is used. Numerical range should be [-pi,pi]
    @return: the unwrapped phases
    """

    dtype = matrix.dtype
    dims = matrix.shape

    if len(dims)>2: raise ValueError("matrix has too many dimensions to unwrap")
    if len(dims) < 2:
        matrix.shape = (1,dims[0])

    if mask is None:
        mask = 255*(N.ones(matrix.shape, N.uint8))
    else:
        mask = N.where(mask, 255, 0).astype(N.uint8)
    if dims != mask.shape:
        raise ValueError("mask dimensions do not match matrix dimensions!")

    ret = Unwrap2D(matrix.astype(N.float32), mask).astype(dtype)
    ret.shape = dims
    return ret

def unwrap3D(matrix):
    """The method for this module unwraps a 3D array of wrapped phases
    using the lp-norm method.
    @param matrix, if ndim > 3, explode; if ndim < 2, a 1xN matrix
    is used. Numerical range should be [-pi,pi]
    @return: the unwrapped phases
    """

    dtype = matrix.dtype
    dims = matrix.shape

    if len(dims)>3:
        raise ValueError("matrix has too many dimensions to unwrap")
    if len(dims) < 3:
        dims = (1,)*(3-len(dims)) + dims

    in_phase = N.reshape(matrix, dims).astype(N.float32)
    
    # actually this seems better pre-masked (confirm??)
    ret = Unwrap3D(in_phase).astype(dtype)
    return N.reshape(ret, matrix.shape)


def test2d():
    from recon.util import normalize_angle
    grid = N.outer(N.ones(64), N.arange(-32,32)) + \
           1.j * N.outer(N.arange(-32,32), N.ones(64))
    pgrid = N.abs(grid)
    wr_grid = normalize_angle(pgrid)
    uw_grid = unwrap2D(wr_grid)
    uw_grid += (pgrid[32,32] - uw_grid[32,32])
    try:
        N.testing.assert_array_almost_equal(pgrid, uw_grid, decimal=5)
        print "passed!"
    except:
        print "2D unwrapping failed easy test!!!"
    
def test3d():
    from recon.util import normalize_angle
    grid = N.empty((64,64,64), N.float32)
    for l in range(64):
        for m in range(64):
            for n in range(64):
                grid[l,m,n] = (l-32)**2 + (m-32)**2 + (n-32)**2
    N.power(grid, 0.5, grid)
    wr_grid = normalize_angle(grid)
    uw_grid = unwrap3D(wr_grid)
    uw_grid += (grid[32,32,32] - uw_grid[32,32,32])
    try:
        N.testing.assert_array_almost_equal(grid, uw_grid, decimal=5)
        print "passed!"
    except:
        print "3D unwrapping failed easy test!!!"
    
