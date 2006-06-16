try:
    from _punwrap import lpUnwrap
except ImportError:
    raise ImportError("Please compile the punwrap extension to use this module")

import Numeric

#masking in _punwrap.so doesn't seem to work well

def unwrap2D(matrix, mask=None):
    """The method for this module unwraps a 2D grid of wrapped phases
    using the lp-norm method.
    @param matrix, if ndim > 2, explode; if ndim < 2, a 1xN matrix
    is used. Numerical range should be [-pi,pi]
    @return: the unwrapped phases
    """

    dtype = matrix.typecode()
    dims = matrix.shape

    if len(dims)>2: raise ValueError("matrix has too many dimensions to unwrap")
    if mask is None: mask = Numeric.ones(dims)
    else:
        #print "warning: masking feature has unpredictable results"
        if dims != mask.shape:
            raise ValueError("mask dimensions do not match matrix dimensions!")
    
    in_phase = len(dims) < 2 and reshape(matrix,(1,dims[0])) or matrix
    #in_phase = ( (in_phase/2/Numeric.pi + 1)%1 ).astype(Numeric.Float32)
    #ret = (lpUnwrap(in_phase, mask.astype(Numeric.Int8))*2*Numeric.pi).astype(dtype)
    in_phase[:] = (in_phase*mask).astype(dtype)
    ret = (lpUnwrap(in_phase.astype(Numeric.Float32)).astype(dtype)
    return ret
