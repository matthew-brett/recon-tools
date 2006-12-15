from pylab import ones, zeros, Float32, Complex32, multiply, pi, \
     angle, conjugate, putmask, Int8, power, diff, average, array, find, median
from Numeric import empty, sum, sort, sign, take, reshape
from LinearAlgebra import *
from recon.punwrap import unwrap2D
from recon.nifti import writeImage
from recon.operations import Operation, Parameter

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
    heights = zeros((shape[:2]), Float32)
    for t in range(shape[0]):
        masks[t] = build_3Dmask(power(vols[t], 0.5))
        for z in range(shape[1]):
            uw_phase[t,z] = unwrap2D(angle(vols[t,z]), mask=masks[t,z])
            #uw_phase[t,z] = angle(vols[t,z])*masks[t,z]
            heights[t,z] = average(array([average(take(row,find(masks[t,z,r])))
                                          for r,row in enumerate(uw_phase[t,z])\
                                          if sum(masks[t,z,r])]))
            heights[t,z] = 2*pi*int((heights[t,z] + sign(heights[t,z])*pi)/2/pi)
        # bring every one down/up to the median height
        heights[t] = (heights[t] - median(heights[t])).astype(Float32)
        uw_phase[t] = uw_phase[t] - reshape(heights[t], (shape[1],1,1))
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
        )
    
    #-------------------------------------------------------------------------
    def run(self, image):

        # Make sure it's an asems image
        if not hasattr(image._procpar, "asym_time") and \
               not hasattr(image._procpar, "te"):
            self.log("No asym_time, can't compute field map.")
            return
        asym_times = image._procpar.get('asym_time', False) or \
                     image._procpar.get('te', False)

        # Make sure there are at least two volumes
        if image.tdim < 2:
            self.log("Cannot calculate field map from only a single volume."\
                     "  Must have at least two volumes.")
            return

        # Unwrap phases.
        diff_vols = zeros((image.tdim-1,image.zdim,image.ydim,image.xdim), \
                          Complex32)
        for vol in range(image.tdim-1):
            diff_vols[vol] = conjugate(image[vol+1])*image[vol]
        phase_map, bytemasks = unwrap_phase(diff_vols)
        for vol in range(image.tdim-1):
            asym_time = asym_times[vol] - asym_times[vol+1]
            phase_map[vol] = (phase_map[vol]/asym_time).astype(Float32)
        fmap_im = image._subimage(phase_map)
        bmask_im = image._subimage(bytemasks)
        # for each diff vol, write a file with vol0 = fmap, vol1 = mask
        for index in range(phase_map.shape[0]):
            catIm = fmap_im.subImage(index).concatenate(
                bmask_im.subImage(index), newdim=True)
            writeImage(catIm, self.fmap_file+"-%d"%index)
