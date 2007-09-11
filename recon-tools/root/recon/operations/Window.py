"Applies a 2D Window (blackman, hamming, or hanning) to the k-space data"
import numpy as N
from recon.operations import Operation, Parameter

window_types = {
    "blackman": N.blackman,
    "hamming": N.hamming,
    "hanning": N.hanning}

def getWindow(winName, xSize, ySize):
    """
    generates a 2D window in following manner:
    outerproduct(window(ySize), window(xSize))
    @param winName: name of the window; can be blackman, hamming, or hanning
    """    
    window = window_types.get(winName)
    if window is None: raise ValueError("unsupported window type: %s"%winName)
    
    p = N.outer(window(ySize), window(xSize))
    
    #return window filter, normalizing just in case
    return p/p.max()


class Window (Operation):
    """
    Apodizes the k-space data based on a specified 2D window.
    """
    params = (
      Parameter(name="win_name", type="str", default="hanning",
        description="""
    Type of window. Can be blackman, hamming, or hanning."""),)

    def run(self, image):
        # multiply the window by each slice of the image
        image *= getWindow(self.win_name, image.idim, image.jdim)
