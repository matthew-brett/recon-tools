"Zero-pads k-space by embedding each NxM slice into a 2Nx2M slice"
import Numeric as N

from recon.operations import Operation
from recon.util import embedIm

class ZeroPad (Operation):
    """
    ZeroPad resizes the data array and shifts the slices down,
    avoiding the creation of a duplicate array
    """
    def run(self, image):
        (nv, ns, ny, nx) = image.tdim and image.shape or (1,) + image.shape
        old_shape = image.shape
        new_shape = list(old_shape)
        new_shape[-2:] = [ny*2, nx*2,]
        image.resize(new_shape)
        #b points to the "old data" inside the resized array, shaped the old way
        b = N.reshape(image[:].flat[0:N.product(old_shape)], old_shape)
        for vol in (nv - N.arange(nv) - 1):
            for sl in (ns - N.arange(ns) - 1):
                if(sl == 0 and vol == 0):
                    continue
                # be careful in case it's a 3d array
                slicer = image.tdim and (vol,sl) or (sl,)
                # put old slicer into new one cornered at (ny/2, nx/2)
                embedIm(b[slicer], image[slicer], ny/2, nx/2)

        #do last slice with permanent copy, or else b gets zero'd
        b = N.reshape(image[:].flat[0:ny*nx], (ny,nx)).copy()
        final_slicer = image.tdim and (0,0) or (0,)
        embedIm(b, image[final_slicer], ny/2, nx/2)
        image.setData(image[:])
        # should set xsize,ysize = xsize/2,ysize/2
    
    #done
