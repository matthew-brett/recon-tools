"Zero-pads k-space by embedding each NxM slice into a 2Nx2M slice"
from imaging.operations import Operation
from pylab import zeros, Complex32, arange, reshape

def embedIm(subIm, Im):
    """
    places subImage into the middle of Image, which is known to have
    dimensions twice as large as subImage (4X area)
    @param subIm: the sub-image
    @param Im: the larger image
    """
    (nSubY, nSubX) = subIm.shape
    #taking for granted that 2*nSubX = nX and nSubX%2 = 0
    yOff = nSubY/2
    xOff = nSubX/2
    Im[:] = zeros((nSubY*2,nSubX*2), Complex32).copy()
    Im[yOff:yOff+nSubY,xOff:xOff+nSubX] = subIm[:,:]
    #done?

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
                embedIm(b[vol,slice], image.data[vol,slice])

        #do last slice with permanent copy, or else b gets zero'd
        b = reshape(image.data.flat[0:ny*nx], (ny,nx)).copy()
        embedIm(b, image.data[0,0])
        image.setData(image.data)
    
    #done
