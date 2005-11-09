import sys
import string 
import os
from Numeric import *
import file_io
import struct
from Numeric import empty
from FFT import inverse_fft
from pylab import pi, mlab, fft, fliplr, zeros, fromstring


#-----------------------------------------------------------------------------
def shift(matrix, axis, shift):
    """
    axis: Axis of shift: 0=x (rows), 1=y (columns), 2=z (slices), etc...
    shift: Number of pixels to shift.
    """
    dims = matrix.shape
    ndim = len(dims)
    if axis >= ndim: raise ValueError("bad axis %s"%axis)
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
    new = empty(dims, matrix.typecode())
    new[tuple(slices_new1)] = matrix[tuple(slices_old1)]
    new[tuple(slices_new2)] = matrix[tuple(slices_old2)]
    matrix[:] = new

#-----------------------------------------------------------------------------
def shifted_fft(a):
    tmp = a.copy()
    shift_width = a.shape[0]/2
    shift(tmp, 0, shift_width)
    tmp = fft(tmp)
    shift(tmp, 0, shift_width)
    return tmp

#-----------------------------------------------------------------------------
def shifted_inverse_fft(a):
    tmp = a.copy()
    shift_width = a.shape[0]/2
    shift(tmp, 0, shift_width)
    tmp = inverse_fft(tmp)
    shift(tmp, 0, shift_width)
    return tmp

#-----------------------------------------------------------------------------
def nice_angle(a): return a + where(a<-pi, 2.*pi, 0) + where(a>pi, -2.*pi, 0)

