try:
    from _punwrap import lpUnwrap
except ImportError:
    raise ImportError("Please compile the punwrap extension to use this module")

import numpy as N

def unwrap2D(matrix, mask=None):
    """The method for this module unwraps a 2D grid of wrapped phases
    using the lp-norm method.
    @param matrix, if ndim > 2, explode; if ndim < 2, a 1xN matrix
    is used. Numerical range should be [-pi,pi]
    @return: the unwrapped phases
    """

    dtype = matrix.dtype
    dims = matrix.shape

    if len(dims)>2: raise ValueError("matrix has too many dimensions to unwrap")
    if mask is None: mask = N.ones(dims)
    else:
        if dims != mask.shape:
            raise ValueError("mask dimensions do not match matrix dimensions!")
    
    in_phase = len(dims) < 2 and N.reshape(matrix,(1,dims[0])) or \
               matrix.copy()
    #in_phase = (in_phase*mask).astype(N.float32)
    in_phase = in_phase.astype(N.float32)
    #ret = (mask*lpUnwrap(in_phase)).astype(dtype)
    ret = lpUnwrap(in_phase).astype(dtype)
    return ret

