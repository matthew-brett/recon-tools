"This is a template module, which can be copied to build new operations"

from imaging.operations import Operation, Parameter
from pylab import reshape, arange, identity, matrixmultiply, inverse

def doNothing(image, fprm, iprm):
    """
    does a lot of nothing to data
    @param fprm: a floating point number
    @param iprm: an integer number
    @return: data, unchanged
    """
    xdim = image.xdim
    ydim = image.ydim
    
    #giving some examples of working with arrays in Python

    #creating an identity matrix
    id = identity(xdim)
    #creating a "column vector" [0 1 2 3 ... xdim-1]^T
    b = reshape(arange(xdim), (xdim, 1))
    #in Python, be careful with your multiplications! If you say:
    #c = id*b
    #meaning to get a copy of b, you'd be surprised to get instead
    #a diagonal matrix whose trace = [0 + 1 + 2 + ... + xdim-1]. What's happened?
    #Python has in fact treated b as an array (which, to be fair, it is)
    #and multiplied it element-wise by the arrays which form the rows in id.
    #If you are familiar with Matlab, this is similar to the .* operation.
    #
    #What I want to say is:
    c = matrixmultiply(id,b)
    #yes, order of parameters is important
    #
    #standard notation for arithmetic is fine for vector/matrix scaling
    #a gain matrix:
    big_id = iprm*id
    #subtracting a DC offset
    c = b - fprm

    #there are also several standard operations in pylab or Numeric, eg:
    little_id = inverse(big_id)

    #can't do it on non-square matrices
    if(ydim != xdim):
        return image.data
    #else return, NOTHING!
    return matrixmultiply(id,image.data)
    #done

class NoOp (Operation):
    """
    A template for operations, with some pointers on Python math
    (does nothing to the data)
    @param fparm: a floating-point number
    @param iparm: an integer
    """

    params=(
        Parameter(name="fparm", type="float", default=0.75,
                  description="A fractional number"),
        Parameter(name="iparm", type="int", default=4,
                  description="A whole number"))

    def run(self, image):
        # do something to image.data here...

        # a Python way of iterating through multiple dimensions
        for vol in image.data:
            for slice in vol:
                image.data = doNothing(image, fparm, iparm)

        # done
        
