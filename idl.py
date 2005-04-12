#!/usr/bic/bin/python
#!/usr/bin/python

import sys
import string
import os
from Numeric import *


#***************************
def shift(matrix,axis,shft):
#***************************

# axis: Axis of shift: 0=x (rows), 1=y (columns),2=z
# shft: Number of pixels to shift.

    dims = matrix.shape
    ndim = len(dims)
    if ndim == 1:
        tmp = zeros((shft),matrix.typecode())
        tmp[:] = matrix[-shft:]
        matrix[-shft:] = matrix[-2*shft:-shft]
        matrix[:shft] = tmp
    elif ndim == 2:
        ydim = dims[0]
        xdim = dims[1]
        tmp = zeros((shft),matrix.typecode())
        new = zeros((ydim,xdim),matrix.typecode())
        if(axis == 0):
            for y in range(ydim):
                tmp[:] = matrix[y,-shft:]
                new[y,shft:] =  matrix[y,:-shft]
                new[y,:shft] = matrix[y,-shft:]
            matrix[:,:] = new[:,:]
        elif(axis == 1):
            for x in range(xdim):
                new[shft:,x] =  matrix[:-shft,x]
                new[:shft,x] = matrix[-shft:,x]
            matrix[:,:] = new[:,:]
    elif ndim == 3:
        zdim = dims[0]
        ydim = dims[1]
        xdim = dims[2]
        new = zeros((zdim,ydim,xdim),matrix.typecode())
        if(axis == 0):
            tmp = zeros((zdim,ydim,shft),matrix.typecode())
            tmp[:,:,:] = matrix[:,:,-shft:]
            new[:,:,shft:] =  matrix[:,:,:-shft]
            new[:,:,:shft] = matrix[:,:,-shft:]
        elif(axis == 1):
            tmp = zeros((zdim,shft,xdim),matrix.typecode())
            tmp[:,:,:] = matrix[:,-shft:,:]
            new[:,shft:,:] =  matrix[:,:-shft,:]
            new[:,:shft,:] = matrix[:,-shft:,:]
        elif(axis == 2):
            tmp = zeros((shft,ydim,xdim),matrix.typecode())
            tmp[:,:,:] = matrix[-shft:,:,:]
            new[shft:,:,:] =  matrix[:-shft,:,:]
            new[:shft,:,:] = matrix[-shft:,:,:]
        matrix[:,:,:] = new[:,:,:]
    else:
        print "shift() only support 1D, 2D, and 3D arrays."
        sys.exit(1)

    return
