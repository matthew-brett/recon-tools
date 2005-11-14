import glob
from os.path import join as pjoin
from pylab import asarray
from FDFFile import FDFFile
from ProcPar import ProcPar, ProcParImageMixin
from imaging.imageio import BaseImage, write_analyze, get_dims


##############################################################################
class FDFImage (BaseImage, ProcParImageMixin):

    #-------------------------------------------------------------------------
    def __init__(self, datadir):
        self.loadParams(datadir)
        self.loadData(datadir)

    #-------------------------------------------------------------------------
    def loadData(self, datadir):
        volumes = []
        self.zdim = len(self._procpar.pss)
        for volnum in self.image_vols:
            slices = []
            for slicenum in range(self.zdim):
                filename = "image%04d.fdf"%(volnum*self.zdim + slicenum + 1)
                slices.append(FDFFile(pjoin(datadir, filename)).data)
            volumes.append(asarray(slices))
        self.data = asarray(volumes)
        self.ndim, self.tdim, self.zdim, self.ydim, self.xdim = \
          get_dims(self.data)

    #-------------------------------------------------------------------------
    def save(self, outputdir):
        for volnum, volimage in enumerate(self.subImages()):
            write_analyze(volimage, pjoin(outputdir, "image%04d"%volnum))
