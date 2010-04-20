"""
This module provides N-D FFTs for functions taken on the interval
n = [-N/2, ..., N/2-1] in all transformed directions. This is accomplished
quickly by making a change of variables in the DFT expression, leading to
multiplication of exp(+/-jPIk) * DFT{exp(+/-jPIn) * [n]}. Take notice that
BOTH your input and output arrays will be arranged on the negative-to-positive
interval. To take regular FFTs, shifting can be turned off.
"""
import numpy as np
from os.path import join, split, abspath
import os, sys
from recon import loads_extension_on_call

nthreads = 1
if sys.platform == 'darwin':
    try:
        nthreads = int(os.popen('sysctl -n hw.activecpu').read().strip())
    except:
        pass
elif sys.platform != 'win32':
    try:
        nthreads = os.sysconf('SC_NPROCESSORS_ONLN')
    except:
        pass
if sys.platform != 'win32':
    def export_extension(build=False):
        from scipy.weave import ext_tools
        from scipy.weave.converters import blitz as blitz_conv
        from numpy.distutils.system_info import get_info

        # call_code needs python args ['a','b','adims','fft_sign','shift','inplace']
        call_code = """
        if(inplace) {
          SCL_TYPE_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
        } else {
          SCL_TYPE_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
        }
        """

        fft_code = open(join(split(__file__)[0],'src/blitz_ffts.cc')).read()

        scl_types = dict(( (np.dtype('F'), 'cfloat'),
                           (np.dtype('D'), 'cdouble') ))

        blitz_ranks = range(1,12)

        fftw_info = get_info('fftw3')
        fftw_libs = ['fftw3', 'fftw3f']
        defines = []

        if nthreads > 1:
            fftw_libs += ['pthread']
            defines += [('THREADED', None),
                        ('NTHREADS', nthreads),
                        ('BZ_THREADSAFE', None)]

        ext_funcs = []
        fft_sign = -1
        shift = 1
        inplace = 0
        for scl_type in scl_types:
            for rank in blitz_ranks:
                shape = (1,) * rank
                a = np.empty(shape, scl_type)
                b = np.empty(shape, scl_type)
                adims = np.array([0], 'i')
                c_scl_type = scl_types[scl_type]
                fcode = call_code.replace('SCL_TYPE', c_scl_type)
                fname = '_fft_%s_%d'%(scl_type.char, rank)

                ext_funcs.append(ext_tools.ext_function(fname, fcode,
                                                        ['a', 'b', 'adims',
                                                         'fft_sign', 'shift',
                                                         'inplace'],
                                                        type_converters=blitz_conv))
        mod = ext_tools.ext_module('fft_ext')
        for func in ext_funcs:
            mod.add_function(func)
        mod.customize.add_support_code(fft_code)
        mod.customize.set_compiler('gcc')
        mod.customize.add_header('<fftw3.h>')
        kw = {'libraries':fftw_libs,
              'include_dirs':fftw_info['include_dirs'] + [np.get_include()],
              'library_dirs':fftw_info['library_dirs'],
              'define_macros':defines}
        loc = split(__file__)[0]
        if build:
            mod.compile(location=loc, **kw)
        else:
            # this also generates the cpp file
            ext = mod.setup_extension(location=loc, **kw)
            ext.name = 'recon.fftmod.fft_ext'
            return ext

#______________________ Some convenience wrappers ___________________________

def fft1(a, shift=True, inplace=False, axis=-1):
    return _fftn(a, axes=(axis,), shift=shift, inplace=inplace)

def ifft1(a, shift=True, inplace=False, axis=-1):
    return _ifftn(a, axes=(axis,), shift=shift, inplace=inplace)

def fft2(a, shift=True, inplace=False, axes=(-2,-1)):
    return _fftn(a, axes=axes, shift=shift, inplace=inplace)    

def ifft2(a, shift=True, inplace=False, axes=(-2,-1)):
    return _ifftn(a, axes=axes, shift=shift, inplace=inplace)
#____________________________________________________________________________

if sys.platform != 'win32':
    @loads_extension_on_call('fft_ext', locals())
    def _fftn(a, axes=(-1,), shift=1, inplace=0, fft_sign=-1):
        # integer-ize these parameters
        shift = int(shift)
        inplace = int(inplace)

        rank = len(a.shape)
        fname = '_fft_%s_%d'%(a.dtype.char, rank)
        try:
            ft_func = getattr(fft_ext, fname)
        except AttributeError:
            raise ValueError('no transform for this type and rank: %s, %d'%(a.dtype.char, rank))

        if inplace:
            # create a very small full rank array b to make ref counts happy
            full_rank = tuple( [1] * rank )
            b = np.array([1], dtype=a.dtype).reshape(full_rank)
        else:
            b = np.empty_like(a)

        adims = np.array(axes, dtype='i')
        ft_func(a, b, adims, fft_sign, shift, inplace)
        if not inplace:
            return b

else:
    def _fftn(a, axes=(-1,), shift=1, inplace=0, fft_sign=-1):
        fft_func = (fft_sign<0) and np.fft.fftn or np.fft.ifftn
        if inplace:
            op_arr = a
        else:
            op_arr = a.copy()
        if shift:
            for n, d in enumerate(a.shape):
                updown = 1 - 2*(np.arange(d)%2)
                slices = [np.newaxis] * len(a.shape)
                slices[n] = slice(None)
                op_arr *= updown[slices]
        #else:
        #    b = a.copy()
        b = fft_func(op_arr, axes=axes)
        del op_arr
        if shift:
            for n, d in enumerate(a.shape):
                updown = 1 - 2*(np.arange(d)%2)
                slices = [np.newaxis] * len(a.shape)
                slices[n] = slice(None)
                b *= updown[slices]
        if inplace:
            a[:] = b
            del b
            return
        return b

def _ifftn(*args, **kwargs):
    kwargs['fft_sign'] = +1
    return _fftn(*args, **kwargs)
    
    
