"Zero-pads k-space by embedding each NxM slice into a 2Nx2M slice"
from imaging.operations import Operation
from pylab import zeros, Complex32, arange, reshape
from imaging.util import embedIm

class ZeroPad (Operation):
    """
    ZeroPad resizes the data array and shifts the slices down,
    avoiding the creation of a duplicate array
    """
    def run(self, image):
        (nv, ns, ny, nx) = (image.tdim, image.zdim, image.ydim, image.xdim)
        image.data.resize((nv,ns,ny*2,nx*2))
        #b points to the "old data" inside the resized array, shaped the old way
        b = reshape(image.data.flat[0:nv*ns*ny*nx], (nv, ns, ny, nx))
        for vol in (nv - arange(nv) - 1):
            for slice in (ns - arange(ns) - 1):
                if(slice == 0 and vol == 0):
                    continue
                # put old slice into new one cornered at (ny/2, nx/2)
                embedIm(b[vol,slice], image.data[vol,slice], ny/2, nx/2)

        #do last slice with permanent copy, or else b gets zero'd
        b = reshape(image.data.flat[0:ny*nx], (ny,nx)).copy()
        embedIm(b, image.data[0,0])
        image.setData(image.data)
    
    #done
