"Applies a 2D Window (blackman, hamming, or hanning) to the k-space data"
from imaging.operations import Operation, Parameter
from pylab import hanning, hamming, blackman, outerproduct, Complex32

window_types = {
    "blackman": blackman,
    "hamming": hamming,
    "hanning": hanning}

def getWindow(winName, xSize, ySize):
    """
    generates a 2D window in following manner:
    outerproduct(window(ySize), window(xSize))
    @param winName: name of the window; can be blackman, hamming, or hanning
    """    
    window = window_types.get(winName)
    if window is None: raise ValueError("unsupported window type: %s"%winName)
    
    p = outerproduct(window(ySize), window(xSize))
    
    #return window filter, normalizing just in case
    return (p/max(p.flat)).astype(Complex32)


class Window (Operation):
    params = (
      Parameter(name="win_name", type="str", default=None,
        description="Type of window.  Can be blackman, hamming, or hanning."),)

    def run(self, image):
        # multiply the window by each slice of the image
        image.data *= getWindow(self.win_name, image.xdim, image.ydim)
