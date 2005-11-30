import sys
import string 
import os
from Numeric import *
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
def normalize_angle(a):
    return a + where(a<-pi, 2.*pi, 0) + where(a>pi, -2.*pi, 0)

#-----------------------------------------------------------------------------
def median_filter(image, N):
    "Filter an image with a median filter of order NxN where N is odd."
    if not N%2: raise ValueError("Order of median filter must be odd.")
    ndim = len(image.shape)
    if ndim < 2:
        raise ValueError("Image dimension must be at least 2 for median_filter")
    tdim, zdim, ydim, xdim = (1,)*(4-ndim) + image.shape[-4:]

    median_pt = int((N*N)/2)
    center = int(N/2)
    image = reshape(image,(tdim*zdim, ydim, xdim))
    img = empty(image.shape, image.typecode())
    subm = empty((N, N), image.typecode())
    for tz in range(tdim*zdim):
        img[tz, 0, :] = 0.
        img[tz, -1:, :] = 0.
        for y in range(ydim-N):
            img[tz, y, 0] = 0.
            img[tz, y, xdim-1] = 0.
            for x in range(xdim-N):
                subm[:,:] = image[tz, y+1:y+N+1, x+1:x+N+1]
                s = sort(subm.flat)
                img[tz, y+center+1, x+center+1] = s[median_pt]
    return reshape(img, (tdim, zdim, ydim, xdim))
