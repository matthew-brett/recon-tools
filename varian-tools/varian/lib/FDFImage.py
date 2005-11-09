import glob
import os
from pylab import asarray
from FDFFile import FDFFile


##############################################################################
class FDFImage (object):

    #-------------------------------------------------------------------------
    def __init__(self, datadir):
        self.datadir = datadir
        self.loadParams()
        self.loadData()

    #-------------------------------------------------------------------------
    def loadParams(self):
        procpar = varian.procpar(os.path.join(self.datadir),"procpar")
        self.tdim = procpar.images[0]
        self.zdim = len(procpar.pss)

    #-------------------------------------------------------------------------
    def loadData(self):
        volumes = []
        for volnum in self.image_vols:
            slices = []
            for slicenum in range(self.zdim)
                filename = "image%04d.fdf"%(volnum*self.zdim + slicenum + 1)
                slices.append(FDFFile(filename).data)
            volumes.append(asarray(slices)
        self.image_data = asarray(volumes)
