import glob
from os.path import join as pjoin
import numpy as N

from recon.scanners.varian.FDFFile import FDFFile, FDFHeader
from ProcPar import ProcPar, ProcParImageMixin
from recon.imageio import ReconImage, writeImage

#-----------------------------------------------------------------------------
def slicefilename(slicenum): return "image%04d.fdf"%slicenum


##############################################################################
class FDFImage (ReconImage, ProcParImageMixin):

    #-------------------------------------------------------------------------
    def __init__(self, datadir):
        self.datadir = datadir
        ProcParImageMixin.__init__(self, datadir)
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
            volumes.append(N.asarray(slices))
        self.setData(N.asarray(volumes))

    #-------------------------------------------------------------------------
    def save(self, outputdir): writeImage(self, pjoin(outputdir, "image"))
