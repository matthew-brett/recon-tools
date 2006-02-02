"This module defines a rotation on all slices, putting them into standard radiological format"

from pylab import zeros, reshape, Complex32, arange, take, ravel, transpose
from imaging.operations import Operation, Parameter
from imaging.imageio import get_dims

#functioning
#TODO: perhaps expand to an arbitrary n*pi/2 rotation?

def rotateIm2d(im2d, rowSize, colSize):
    """
    This method simply takes a 2D matrix and rotates it +90 degrees (equivalent
    to an upside-down transpose).
    @param im2d: a 2D MxN matrix
    @param rowSize: := M (number of rows)
    @param colSize: := N (number of columns)
    @returns: a 2D NxM matrix (for now)
    """

    twist = zeros((colSize,rowSize), im2d.typecode())
    rev = -1 - arange(colSize)
    for row in range(rowSize):
        #this is good for -90 degrees: 
        #twist[:,-(row+1)] = reshape(take(im2d[row], rev), (colSize,1))[:,0]

        #this is good for +90 degrees:
        twist[:,row] = reshape(take(im2d[row], rev), (colSize,1))[:,0]

        #this may be good for 180 degrees (untested, assumes logic to shape twist correctly):
        #twist[-(row+1),:] = take(im2d[row], rev)[0,:]
    return twist


class Rot90 (Operation):
    """
    This class operatates slice-by-slice to rotate the images so that they are in
    standard radiological format.
    """

    def run(self, image):
        #data dimensions haven't been set yet?
        image.setData(image.data)
        (nvols, nslices, nrows, ncols) = (image.tdim, image.zdim, image.ydim, image.xdim)        
        for vol in image.data:
            for slice in vol:
                #nrows,ncols may be backwards, I'm still confused why nrows = y, ncols = x
                slice.flat[:] = rotateIm2d(slice, nrows, ncols).flat

        #need to swap the way Python indexes the array  
        #this might be changed when there is something callable for a BaseImage:
        image.data = reshape(image.data, (nvols, nslices, ncols, nrows))
        image.setData(image.data)
                
    #done
