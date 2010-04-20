import numpy as np
from recon import loads_extension_on_call

dtype2ctype = {
    np.dtype(np.complex128): 'std::complex<double>',
    np.dtype(np.complex64): 'std::complex<float>',
    np.dtype(np.float64): 'double',
    np.dtype(np.float32): 'float',
    np.dtype(np.int32): 'long',
    np.dtype(np.int16): 'short',
    np.dtype(np.uint16): 'unsigned short',
    np.dtype(np.uint8): 'unsigned char',
    }

def compose_xform(mat, view=True, square=True, prev_xform=None):
    """This composes a transformation made up of transpose and reflection
    steps.

    arguments
    mat : 2, or 3D (i,j,[k]) --> (x,y,[z]) transform

    The optional arguments are:
    view : if view is True, don't use any memory changing algorithms
    square : if the array is square in-plane, the inplace xpose is used
    prev_xform : any data transform to apply a prior
    """
    sensible = len(mat.shape)==2 and mat.shape[0]==mat.shape[1]
    assert sensible, "transformation matrix is not full rank:"+str(mat)
    nd = mat.shape[0]
    if nd > 2 and not mat[2,2]:
        # want to change this to: if not mat.diagonal().any()
        raise ValueError("""Can't perform this transform\n"""+str(mat))
    if view or not square:
        xpose = np.swapaxes
    else:
        xpose = square_xpose
    xform = prev_xform or (lambda x: x)
    if view:
        # a view only, don't mess with the memory
        xyz_reflect = [lambda x: reverse(x, axis=-1),
                       lambda x: reverse(x, axis=-2),
                       lambda x: reverse(x, axis=-3)]
    else:
        # do the in-place algorithms
        xyz_reflect = [reverse_x, reverse_y, reverse_z]
    (X,Y,Z) = [0,1,2]
    if not mat[0,0]:
        # if i and j are not colinear with x and y, transpose the plane.
        xform = lambda x, g=xform: xpose(g(x), -1, -2)
    # now that i,j have been aligned with (x,y), do any indicated reflections
    if (mat[0,0] < 0 or mat[0,1] < 0):
        # need to flip +x -> -x
        xform = lambda x, g=xform: xyz_reflect[X](g(x))
    if (mat[1,0] < 0 or mat[1,1] < 0):
        # need to flip +y -> -y
        xform = lambda x, g=xform: xyz_reflect[Y](g(x))
    if nd > 2 and mat[2,2] < 0:
        xform = lambda x, g=xform: xyz_reflect[Z](g(x))
    return xform


def _xform_factory(fname):
    @loads_extension_on_call('inplane_xforms_x', globals())
    def anon(M, *args):
        dshape = M.shape
        M.shape = (1,)*(4-len(dshape)) + dshape
        
        func = eval('inplane_xforms_x.'+fname+'_'+M.dtype.char)
        func(M, *M.shape)
        
        M.shape = dshape
        return M
    anon.__name__ = fname
    return anon
        
square_xpose = _xform_factory('square_xpose')
reverse_x = _xform_factory('reverse_x')
reverse_y = _xform_factory('reverse_y')
reverse_z = _xform_factory('reverse_z')

def reverse(a, axis=-1):
    slices = [slice(0,d) for d in a.shape]
    slices[axis] = slice(-1, -a.shape[axis]-1, -1)
    return a[tuple(slices)]

def export_extension(build=False):
    from scipy.weave import ext_tools
    from scipy.weave.converters import blitz
    from os.path import join, split

    xpose_code = """
    int t, i, j, k;
    %s tmp;
    
    for(t=0; t<tdim; t++) {
      for(k=0; k<kdim; k++) {
        for(j=0; j<jdim; j++) {
          for(i=0; i<j; i++) {
            tmp = M(t,k,i,j);
            M(t,k,i,j) = M(t,k,j,i);
            M(t,k,j,i) = tmp;
          }
        }
      }
    }
    """
    revx_code = """
    int t, i, j, k;
    %s tmp;
    for(t=0; t<tdim; t++) {
      for(k=0; k<kdim; k++) {
        for(j=0; j<jdim; j++) {
          for(i=0; i<idim/2; i++) {
            tmp = M(t,k,j,idim-i-1);
            M(t,k,j,idim-i-1) = M(t,k,j,i);
            M(t,k,j,i) = tmp;
          }
        }
      }
    }
    """
    revy_code = """
    int t, i, j, k;
    %s tmp;
    for(t=0; t<tdim; t++) {
      for(k=0; k<kdim; k++) {
        for(i=0; i<idim; i++) {
          for(j=0; j<jdim/2; j++) {
            tmp = M(t,k,jdim-j-1,i);
            M(t,k,jdim-j-1,i) = M(t,k,j,i);
            M(t,k,j,i) = tmp;
          }
        }
      }
    }
    """
    revz_code = """
    int t, i, j, k;
    %s tmp;
    for(t=0; t<tdim; t++) {
      for(j=0; j<jdim; j++) {
        for(i=0; i<idim; i++) {
          for(k=0; k<kdim/2; k++) {
            tmp = M(t,kdim-k-1,j,i);
            M(t,kdim-k-1,j,i) = M(t,k,j,i);
            M(t,k,j,i) = tmp;
          }
        }
      }
    }
    """
    codes = {
        'square_xpose': xpose_code,
        'reverse_x': revx_code,
        'reverse_y': revy_code,
        'reverse_z': revz_code,
        }

    mod = ext_tools.ext_module('inplane_xforms_x')
    for func, code in codes.items():
        for dt, ct in dtype2ctype.items():
           func_name = func + '_' + dt.char
           func_code = code%ct
           locals_dict = {}
           locals_dict['M'] = np.empty((1,1,1,1), dt)
           locals_dict.update(dict(idim = 1, jdim = 1,
                                   kdim = 1, tdim = 1))
           modfunc = ext_tools.ext_function(func_name, func_code,
                                            ['M','tdim','kdim','jdim','idim'],
                                            type_converters=blitz,
                                            local_dict=locals_dict)
           mod.add_function(modfunc)
    mod.customize.set_compiler('gcc')
    loc = split(__file__)[0]
    kw = {'include_dirs': [np.get_include()]}
    if build:
        mod.compile(location=loc, **kw)
    else:
        ext = mod.setup_extension(location=loc, **kw)
        ext.name = 'recon.inplane_xforms_x'
        return ext

