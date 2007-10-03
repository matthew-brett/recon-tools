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

    
