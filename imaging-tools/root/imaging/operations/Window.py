"Applies a 2D Window (blackman, hamming, or hanning) to the k-space data"
from imaging.operations import Operation, Parameter
from pylab import hanning, hamming, blackman, outerproduct, Complex32

def getWindow(winName, xSize, ySize):
    """
    generates a 2D window in following manner:
    outerproduct(window(ySize), window(xSize))
    @param winName: name of the window; can be blackman, hamming, or hanning
    """    

    #actually gets a KeyError on a bad winName, should fix later
    window = {
        "blackman": blackman,
        "hamming": hamming,
        "hanning": hanning
    }[winName]
    if winName is None:
        raise "unsupported window type: %s"%winName
    
    p = outerproduct(window(ySize), window(xSize))
    
    #return window filter, normalizing just in case
    return (p/max(p.flat)).astype(Complex32)


class Window (Operation):
    params = (
        Parameter(name="win_name", type="str", default=None,
                  description="name of desired window"),
        )

    def run(self, image):
        #this will be killed in some future revision:
        image.setData(image.data)
        #start here:
        winFilt = getWindow(self.win_name, image.xdim, image.ydim)
        for vol in image.data:
            for slice in vol:
                slice[:] = slice*winFilt

        #done
