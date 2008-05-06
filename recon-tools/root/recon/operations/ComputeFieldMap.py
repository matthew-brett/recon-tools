import numpy as N

from recon.punwrap import unwrap3D
from recon.operations import Operation, Parameter

def build_3Dmask(vol, threshfactor):
    mask = N.ones(vol.shape, N.float32)
    p = N.sort(vol.flatten())
    npts = p.shape[0]
    t2 = p[int(round(.02*npts))]
    t98 = p[int(round(.98*npts))]
    thresh = threshfactor*(t98 - t2) + t2
    N.putmask(mask, vol<thresh, 0)
    return mask

##############################################################################
class ComputeFieldMap (Operation):
    """
    Perform phase unwrapping and calculate B0 inhomogeneity field map.
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
        if not hasattr(image, "asym_times") and not hasattr(image, "te"):
            self.log("No asym_time, can't compute field map.")
            return
        asym_times = image.asym_times

        # Make sure there are at least two volumes
        if image.tdim < 2:
            self.log("Cannot calculate field map from only a single volume."\
                     "  Must have at least two volumes.")
            return

        diffshape = (image.tdim-1,) + image.shape[-3:]
        diffshape = image.tdim > 2 and diffshape or diffshape[-3:]
        diff_vols = image._subimage(N.empty(diffshape, N.complex64))
        phase_map = image._subimage(N.empty(diffshape, N.float32))
        bytemask = image._subimage(N.empty(diffshape, N.float32))
        # Get phase difference between scan n+1 and scan n
        # Then find the mask and unwrapped phase, and compute
        # the field strength in terms of rad/s
        for dsub, psub, msub in zip(diff_vols, phase_map, bytemask):
            del_te = asym_times[dsub.num+1] - asym_times[dsub.num]
            dsub[:] = image[dsub.num+1] * N.conjugate(image[dsub.num])
            msub[:] = build_3Dmask(N.power(abs(dsub[:]), 0.5),
                                   self.threshfactor)
            psub[:] = (unwrap3D(N.angle(dsub[:])) * msub[:]) / del_te

        # for each diff vol, write a file with vol0 = fmap, vol1 = mask
        if phase_map.ndim > 3:
            for index in range(phase_map.tdim):
                catIm = phase_map.subImage(index).concatenate(
                    bytemask.subImage(index), newdim=True)
                catIm.writeImage(self.fmap_file+"-%d"%index,
                                 format_type="nifti-single")
        else:
            catIm = phase_map.concatenate(bytemask, newdim=True)
            catIm.writeImage(self.fmap_file, format_type="nifti-single")
