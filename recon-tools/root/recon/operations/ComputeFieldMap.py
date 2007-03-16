import numpy as N

from recon.punwrap import unwrap2D
from recon.operations import Operation, Parameter

def build_3Dmask(vol, threshfactor):
    mask = N.ones(vol.shape, N.int8)
    p = N.sort(abs(vol.flatten()))
    t2 = p[int(round(.02*len(p)))]
    t98 = p[int(round(.98*len(p)))]
    thresh = threshfactor*(t98 - t2) + t2
    N.putmask(mask, abs(vol)<thresh, 0)
    return mask

def unwrap_phase(vols, threshfactor):
    shape = vols.shape
    uw_phase = N.empty(shape, N.float32)
    masks = N.empty(shape, N.int8)
    heights = N.zeros((shape[:2]), N.float32)
    for t in range(shape[0]):
        masks[t] = build_3Dmask(N.power(vols[t], 0.5), threshfactor)
        for z in range(shape[1]):
            uw_phase[t,z] = unwrap2D(N.angle(vols[t,z]), mask=masks[t,z])
            #uw_phase[t,z] = N.angle(vols[t,z])*masks[t,z]
            heights[t,z] = N.array([N.take(row,N.nonzero(masks[t,z,r])).mean()
                                    for r,row in enumerate(uw_phase[t,z]) \
                                    if N.sum(masks[t,z,r])]).mean()
            heights[t,z] = 2*N.pi*int((heights[t,z] + N.sign(heights[t,z])*N.pi)/2/N.pi)
        # bring every one down/up to the median height
        heights[t] = (heights[t] - N.median(heights[t])).astype(N.float32)
        uw_phase[t] = uw_phase[t] - N.reshape(heights[t], (shape[1],1,1))
    return uw_phase, masks

##############################################################################
class ComputeFieldMap (Operation):
    """
    Perform phase unwrapping and calculate phase-warping field map.
    """

    params=(
        Parameter(name="fmap_file", type="str", default="fieldmap",
                  description="""
    Name of the field map file to store"""),
        Parameter(name="threshfactor", type="float", default=0.1,
                  description="""
    Adjust this factor to raise or lower the masking threshold for the
    ASEMS signal."""),)
    
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
        diffshape = (image.tdim-1, image.zdim, image.ydim, image.xdim)
        diff_vols = N.zeros(diffshape, N.complex128)
        for vol in range(image.tdim-1):
            diff_vols[vol] = N.conjugate(image[vol+1])*image[vol]
        phase_map, bytemasks = unwrap_phase(diff_vols, self.threshfactor)
        for vol in range(image.tdim-1):
            asym_time = asym_times[vol] - asym_times[vol+1]
            phase_map[vol] = (phase_map[vol]/asym_time).astype(N.float32)
        fmap_im = image._subimage(phase_map)
        bmask_im = image._subimage(bytemasks)
        # for each diff vol, write a file with vol0 = fmap, vol1 = mask
        for index in range(phase_map.shape[0]):
            catIm = fmap_im.subImage(index).concatenate(
                bmask_im.subImage(index), newdim=True)
            catIm.writeImage(self.fmap_file+"-%d"%index,
                             format_type="nifti-single")
