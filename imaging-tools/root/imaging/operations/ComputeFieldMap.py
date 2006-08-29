from pylab import ones, zeros, Float32, Complex32, multiply, pi, \
     angle, conjugate, putmask, Int8
from Numeric import empty, sum, sort
from LinearAlgebra import *
#from imaging.imageio import writeImage
from imaging.punwrap import unwrap2D
from imaging.nifti import writeImage
from imaging.operations import Operation, Parameter

def build_3Dmask(vol):
    mask = ones(vol.shape, Int8)
    p = sort(abs(vol.flat))
    t2 = p[int(round(.02*len(p)))]
    t98 = p[int(round(.98*len(p)))]
    thresh = 0.1*(t98 - t2) + t2
    putmask(mask, abs(vol)<thresh, 0)
    return mask

def unwrap_phase(vols):
    shape = vols.shape
    uw_phase = empty(shape, Float32)
    masks = empty(shape, Int8)
    for t in range(shape[0]):
        masks[t] = build_3Dmask(vols[t])
        for z in range(shape[1]):
            uw_phase[t,z] = unwrap2D(angle(vols[t,z]), mask=masks[t,z])
    return uw_phase, masks

#-----------------------------------------------------------------------------
def phase_offset(phase):
    nvoxels = multiply.reduce(phase.shape)
    return sum(phase.flat)/(nvoxels*2*pi)

##############################################################################
class ComputeFieldMap (Operation):
    "Perform phase unwrapping and calculate field map"

    params=(
        Parameter(name="fmap_file", type="str", default="fieldmap",
                  description="Name of the field map file to store"),
        Parameter(name="mask_file", type="str", default="volmask",
                  description="Name of the volume mask file to store")
        )
    
    #-------------------------------------------------------------------------
    def run(self, image):

        # Make sure it's an asems image
        if not hasattr(image._procpar, "asym_time"):
            self.log("No asym_time, can't compute field map.")
            return
        asym_times = image._procpar.asym_time

        # Make sure there are at least two volumes
        if image.tdim < 2:
            self.log("Cannot calculate field map from only a single volume."\
                     "  Must have at least two volumes.")
            return

        # Unwrap phases.
        diff_vols = zeros((image.tdim-1,image.zdim,image.ydim,image.xdim), \
                          Complex32)
        for vol in range(image.tdim-1):
            diff_vols[vol] = conjugate(image.data[vol+1])*image.data[vol]
        phase_map, bytemasks = unwrap_phase(diff_vols)
        for vol in range(image.tdim-1):
            asym_time = asym_times[vol] - asym_times[vol+1]
            phase_map[vol] = (phase_map[vol]/asym_time).astype(Float32)
        fmap_im = image._subimage(phase_map)
        bmask_im = image._subimage(bytemasks)
        for index in range(fmap_im.tdim):
            writeImage(fmap_im.subImage(index), self.fmap_file+"-%d"%(index))
            writeImage(bmask_im.subImage(index), self.mask_file+"-%d"%(index))

