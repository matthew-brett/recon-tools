"This is a template module, which can be copied to build new operations"

#This file name is Template.py, found under root/recon/operations

#here I'm importing various definitions from outside of Python:	
from recon.operations import Operation, Parameter
#import the entire Numeric package, renamed as N for convenience
import numpy as N

#I can define a helper function outside of run(image) if it makes
#sense to do so:

def doNothing(slice, fprm, iprm, xSize, ySize):
    """
    does a lot of nothing to data
    @param slice: the 2D image
    @param fprm: a floating point number
    @param iprm: an integer number
    @param xSize: # of columns in a slice
    @param ySize: # of rows in a slice
    @return: slice, unchanged
    """
   
    #giving some examples of working with arrays in Python
    #creating an identity matrix (Numeric functions are accessed as
    #members of the package, which was imported as N)
    id = N.identity(xSize)
    #creating a "column vector" [0 1 2 3 ... xdim-1]^T
    b = N.reshape(N.arange(xSize), (xSize, 1))
    #in Python, be careful with your multiplications! If you say:
    #c = id*b
    #meaning to get a copy of b, you'd be surprised to get instead
    #a diagonal matrix whose trace = [0 + 1 + 2 + ... + xdim-1].
    #What's happened?
    #Python has in fact treated b as an array (which, to be fair, it is)
    #and multiplied it element-wise by the arrays which form the rows in id.
    #If you are familiar with Matlab, this is similar to the .* operation.
    #
    #What I want to say is:
    c = N.dot(id,b)
    #yes, order of parameters is important
    #
    #standard notation for arithmetic is fine for vector/matrix scaling
    #a gain matrix:
    big_id = iprm*id
    #subtracting a DC offset
    c = b - fprm
	
    #there are also several standard operations in pylab or Numeric,
    #eg "inverse" (which is part of LinearAlgebra, provided by Numeric)
    little_id = N.linalg.inv(big_id)
	
    #can't do it on non-square matrices
    if(ySize != xSize):
        return slice
    #else return, NOTHING!
    return N.dot(id,slice)
    #done

#the following line declares a class, and from what class it inherits:	
class Template (Operation):
    """
    A template for operations, with some pointers on Python math
    (does nothing to the data)
    @param fparm: a floating-point number
    @param iparm: an integer
    """

    #the Parameter objects in params are all constructed with four
    #elements: Parameter(name, type, default, description).
    #If you're not using any Parameters, skip this step

    #This params list is actually a Python "tuple". Examples:
    # 3-tuple: (a, b, c); 2-tuple: (a, b); 1-tuple: (a,)
    #The notation is not 100% obvious: be careful to add the
    #trailing comma when defining only 1 Parameter!

    params=(
        Parameter(name="fparm", type="float", default=0.75,
                  description="A fractional number"),
        Parameter(name="iparm", type="int", default=4,
                  description="A whole number"))

    #note the definition of run: the declaration MUST be this way 
    def run(self, image):
        # do something to the ReconImage "image" here...

        #Numeric arrays are indexed in C-order. For a time-series
        #of volumes, image[t,z,y,x] is the voxel at point (x,y,z,t)
        #Be aware that this is the opposite of MATLAB indexing.
        
        #So, the length of the y-dim and x-dim are always the last 2 dimensions
        (ySize, xSize) = image.shape[-2:]
        # a Python way of iterating (secretly using ReconImage's __iter__)
        for vol in image:
            #every "vol" in this iteration is a 3D DataChunk object;
            #a DataChunk also supports __iter__ and can slice into the data
            for slice in vol:
                #every slice is a 2D DataChunk
                slice[:] = doNothing(slice[:], fparm, iparm, xSize, ySize)

	# here the data is being changed in-place
        # final note: the operation returns nothing
        # done
