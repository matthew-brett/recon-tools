from pylab import ones, zeros, Float32, Complex32, multiply, pi, \
     angle, conjugate, putmask
from Numeric import empty, sum, sort
from LinearAlgebra import *
import math
from imaging.imageio import writeImage
from imaging.punwrap import unwrap2D
from imaging.analyze import writeImage
from imaging.operations import Operation

def build_3Dmask(vol):
    mask = ones(vol.shape)
    p = sort(abs(vol.flat))
    t2 = p[int(round(.02*len(p)))]
    t98 = p[int(round(.98*len(p)))]
    thresh = 0.1*(t98 - t2) + t2
    putmask(mask, abs(vol)<thresh, 0)
    return mask

def unwrap_phase(vols):
    shape = vols.shape
    uw_phase = empty(shape, Float32)
    for t in range(shape[0]):
        mask = build_3Dmask(vols[t])
        for z in range(shape[1]):
            uw_phase[t,z] = unwrap2D(angle(vols[t,z]), mask=mask[z])
    return uw_phase

#-----------------------------------------------------------------------------
def phase_offset(phase):
    nvoxels = multiply.reduce(phase.shape)
    return sum(phase.flat)/(nvoxels*2*pi)

##############################################################################
class ComputeFieldMap (Operation):
    "Perform phase unwrapping and calculate field map"

    #-------------------------------------------------------------------------
    def run(self, image):

        # Make sure it's an asems image
        if not hasattr(image._procpar, "asym_time"):
            self.log("No asym_time, can't compute field map.")
            return
        asym_time = image._procpar.asym_time[1]

        # Make sure there are at least two volumes
        if image.tdim < 2:
            self.log("Cannot calculate field map from only a single volume."\
                     "  Must have at least two volumes.")
            return

        # Unwrap phases.
        diff_vols = zeros((image.tdim-1,image.zdim,image.ydim,image.xdim), \
                          Complex32)
        for vol in range(image.tdim-1):
            #diff_vols[vol] = conjugate(image.data[vol])*(image.data[vol+1])
            diff_vols[vol] = conjugate(image.data[vol+1])*image.data[vol]
        phase_map = unwrap_phase(diff_vols)
        phase_map = (phase_map/asym_time).astype(Float32)
        fmap_phase = image._subimage(phase_map)
        for index, subIm in enumerate(fmap_phase.subImages()):
            writeImage(subIm, "fieldmap-%d"%(index))


## FUTURE DEVELOPMENT HERE
##         unwrapped = unwrap_phase(image)
##         unwrapped_vols = list(unwrapped.subImages())
##         for index, phasevol in enumerate(unwrapped_vols[1:]):
##             mask1 = unwrapped_vols[index].data.copy()
##             mask2 = phasevol.data.copy()
##             putmask(mask1, abs(mask1)<1e-6, 0)
##             putmask(mask1, mask1!=0, 1)
##             putmask(mask2, abs(mask2)<1e-6, 0)
##             putmask(mask2, mask2!=0, 1)
##             fmask = mask1*mask2
##             fmap = (phasevol.data - unwrapped_vols[index].data)*fmask
##             offset = round(sum(fmap.flat)/(sum(fmask.flat)*2*pi))
##             if offset != 0:
##                 fmap -= ((2*pi*offset)*fmask).astype(Float32)
##             fmap = (fmap/asym_time).astype(Float32)
##             fmap_image = phasevol._subimage(fmap)
##             writeImage(fmap_image, "fieldmap-%d"%(index))
            
