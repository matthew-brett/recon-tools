import warnings
import numpy as np
from recon import loads_extension_on_call

@loads_extension_on_call('_punwrap2D', locals())
def unwrap2D(matrix, mask=None):
    """
    The method for this module unwraps a 2D grid of wrapped phases
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
        mask = 255*(np.ones(matrix.shape, np.uint8))
    else:
        mask = np.where(mask, 255, 0).astype(np.uint8)
    if dims != mask.shape:
        raise ValueError("mask dimensions do not match matrix dimensions!")

    ret = _punwrap2D.Unwrap2D(matrix.astype(np.float32), mask).astype(dtype)
    ret.shape = dims
    return ret

@loads_extension_on_call('_punwrap3D', locals())
def unwrap3D(matrix):
    """
    The method for this module unwraps a 3D array of wrapped phases
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

    in_phase = np.reshape(matrix, dims).astype(np.float32)
    
    # actually this seems better pre-masked (confirm??)
    ret = _punwrap3D.Unwrap3D(in_phase).astype(dtype)
    return np.reshape(ret, matrix.shape)

    
