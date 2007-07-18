import numpy as N

from recon.punwrap import unwrap3D
from recon.operations import Operation, Parameter

from recon.util import unwrap1D, shift

def build_3Dmask(vol, threshfactor):
    mask = N.ones(vol.shape, N.float32)
    p = N.sort(vol.flatten())
    t2 = p[int(round(.02*len(p)))]
    t98 = p[int(round(.98*len(p)))]
    thresh = threshfactor*(t98 - t2) + t2
    N.putmask(mask, vol<thresh, 0)
    return mask

def unwrap_3Dphase(vols, threshfactor):
    shape = vols.shape
    uw_phase = N.empty(shape, N.float32)
    masks = N.empty(shape, N.float32)
    for t in range(shape[0]):
        wr_phase = N.angle(vols[t])
        masks[t] = build_3Dmask(N.power(abs(vols[t]), 0.5), threshfactor)
        uw_phase[t] = unwrap3D(wr_phase)            
    return uw_phase*masks, masks

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
        diff_vols = N.zeros(diffshape, N.complex64)
        for vol in range(image.tdim-1):
            diff_vols[vol] = N.conjugate(image[vol])*image[vol+1]
        phase_map, bytemasks = unwrap_3Dphase(diff_vols, self.threshfactor)
        for vol in range(image.tdim-1):
            asym_time = asym_times[vol+1] - asym_times[vol]
            #asym_time = 1.0
            phase_map[vol] /= asym_time
        fmap_im = image._subimage(phase_map)
        bmask_im = image._subimage(bytemasks)
        # for each diff vol, write a file with vol0 = fmap, vol1 = mask
        for index in range(phase_map.shape[0]):
            catIm = fmap_im.subImage(index).concatenate(
                bmask_im.subImage(index), newdim=True)
            catIm.writeImage(self.fmap_file+"-%d"%index,
                             format_type="nifti-single")
