import glob
from os.path import join as pjoin
from pylab import asarray
from FDFFile import FDFFile, FDFHeader
from ProcPar import ProcPar, ProcParImageMixin
from imaging.imageio import BaseImage
from imaging.analyze import writeImage


#-----------------------------------------------------------------------------
def slicefilename(slicenum): return "image%04d.fdf"%slicenum


##############################################################################
class FDFImage (BaseImage, ProcParImageMixin):

    #-------------------------------------------------------------------------
    def __init__(self, datadir):
	self.datadir = datadir
        self.loadParams(datadir)
        self.loadData()
 
    #-------------------------------------------------------------------------
    def loadDimSizes(self):
        "@return (xsize, ysize, zsize)"
        if not (self.image_vols or self.zdim): return (0.,0.,0.)
        header = FDFHeader(file(pjoin(self.datadir, slicefilename(1))))
        x, y, z = header.roi
        xpix, ypix = header.matrix
        return ( x/xpix, y/ypix, z )

    #-------------------------------------------------------------------------
    def loadData(self):
        volumes = []
        self.zdim = len(self._procpar.pss)
        for volnum in self.image_vols:
            slices = []
            for slicenum in range(self.zdim):
                filename = slicefilename(volnum*self.zdim + slicenum + 1)
                slices.append(FDFFile(pjoin(self.datadir, filename)).data)
            volumes.append(asarray(slices))
        self.setData(asarray(volumes))

    #-------------------------------------------------------------------------
    def save(self, outputdir):
        for volnum, volimage in enumerate(self.subImages()):
            writeImage(volimage, pjoin(outputdir, "image%04d"%volnum))
