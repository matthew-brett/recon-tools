from pylab import zeros, reshape, Complex32, arange, take, ravel, transpose
from imaging.operations import Operation, Parameter

#functioning
#TODO: change n_fe, n_pe variables ( I think n_pe = y, n_fe = x, ask Brian )
#      perhaps expand to an arbitrary n*pi/2 rotation?

def rotateIm2d(im2d):
    """
    This method simply takes a 2D matrix and rotates it +90 degrees (equivalent
    to an upside-down transpose).
    @param im2d: a 2D MxN matrix
    @returns: a 2D NxM matrix
    """

    (xsize, ysize) = im2d.shape
    twist = zeros((ysize,xsize), im2d.typecode())
    rev = -1 - arange(ysize)
    for row in range(xsize):
        #this is good for -90 degrees: 
        #twist[:,-(row+1)] = reshape(take(im2d[row], rev), (ysize,1))[:,0]

        #this is good for +90 degrees:
        twist[:,row] = reshape(take(im2d[row], rev), (ysize,1))[:,0]

        #this may be good for 180 degrees (untested, assumes logic to shape twist correctly):
        #twist[-(row+1),:] = take(im2d[row], rev)[0,:]
    return twist


class Rot90 (Operation):
    """
    This class operatates slice-by-slice to rotate the images so that they are in
    standard radiological format.
    """

    def run(self, image):
        (nvol, nslices, n_pe, n_fe) = image.data.shape
                
        for vol in image.data:
            for slice in vol:
                slice.flat[:] = rotateIm2d(slice).flat
        image.data = reshape(image.data, (nvol, nslices, n_fe, n_pe))
                
    #done
