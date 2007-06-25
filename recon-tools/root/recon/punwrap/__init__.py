try:
    from _punwrap import Unwrap
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

    # mark this for in_phase = this if True else that (for Python 2.5)
    if len(dims) < 2:
        in_phase = N.reshape(matrix, (1,dims[0]))
    else:
        in_phase = matrix.copy()
    # actually this seems better pre-masked (confirm??)
    in_phase = (mask*in_phase).astype(N.float32)
    ret = Unwrap(in_phase).astype(dtype)
    return ret

