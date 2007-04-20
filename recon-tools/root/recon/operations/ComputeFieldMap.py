import numpy as N

from recon.punwrap import unwrap2D
from recon.operations import Operation, Parameter

from recon.util import unwrap1D, shift

def build_3Dmask(vol, threshfactor):
    mask = N.ones(vol.shape, N.int8)
    p = N.sort(abs(vol.flatten()))
    t2 = p[int(round(.02*len(p)))]
    t98 = p[int(round(.98*len(p)))]
    thresh = threshfactor*(t98 - t2) + t2
    N.putmask(mask, abs(vol)<thresh, 0)
    return mask

def unwrap_3Dphase(vols, threshfactor):
    shape = vols.shape
    uw_phase = N.empty(shape)
    masks = N.empty(shape)
    pe_ref, fe_ref, jmp = shape[-2]/2, shape[-1]/2, 1
    for t in range(shape[0]):
        wr_phase = N.angle(vols[t])
        masks[t] = build_3Dmask(N.power(abs(vols[t]), 0.5), threshfactor)
        zref_pvt = -1
        for z in range(shape[1]):
            uw_phase[t,z] = unwrap2D(wr_phase[z])
            # Want to look for a slice that had no wraps in the raw phase.
            # Since unwrapping may have introduced phase, choose a slice
            # where the difference between wrapped and unwrapped is
            # constant (if not zero).
            if zref_pvt < 0:
                df = masks[t,z]*(uw_phase[t,z] - wr_phase[z])
                dfmax = df.max()
                dfmin = df.min()
                f = abs(dfmax) > abs(dfmin) and dfmax or dfmin
                df_sub = masks[t,z]*abs(df - f)
                if (df_sub < 1e-4).all():
                    zref_pvt = z

        if zref_pvt < 0:
            raise ValueError("All slices have phase wraps, it may not be "\
                             "possible to determine an absolute phase level")
        # Now unwrap each phase surface along a good point in the z-direction,
        # starting at a reference point which we've determined not to have
        # been wrapped. Remember to back-adjust in case the original surface
        # unwrapping created any constant N2PI level difference in that slice.
        wraps = N.zeros(shape[-3])
        while not masks[t,:,pe_ref,fe_ref].all():
            pe_ref += -(N.sign(jmp)*1 + jmp)
            if pe_ref < 0 or pe_ref >= shape[-2]:
                raise IndexError("no through-slice vector found to unwrap")
        zref = uw_phase[t,:,pe_ref,fe_ref].copy()
        shift(zref, 0, -zref_pvt)
        wraps[1:] = unwrap1D(zref, return_diffs=True)
        shift(wraps, 0, zref_pvt) 
        wraps -= (zref[0] - wr_phase[zref_pvt, pe_ref, fe_ref])       
        uw_phase[t][:] = uw_phase[t] + wraps[:,None,None]
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
        diff_vols = N.zeros(diffshape, N.complex128)
        for vol in range(image.tdim-1):
            diff_vols[vol] = N.conjugate(image[vol+1])*image[vol]
        phase_map, bytemasks = unwrap_3Dphase(diff_vols, self.threshfactor)
        for vol in range(image.tdim-1):
            asym_time = asym_times[vol] - asym_times[vol+1]
            #phase_map[vol] = (phase_map[vol]/asym_time).astype(N.float32)
        fmap_im = image._subimage(phase_map)
        bmask_im = image._subimage(bytemasks)
        # for each diff vol, write a file with vol0 = fmap, vol1 = mask
        for index in range(phase_map.shape[0]):
            catIm = fmap_im.subImage(index).concatenate(
                bmask_im.subImage(index), newdim=True)
            catIm.writeImage(self.fmap_file+"-%d"%index,
                             format_type="nifti-single")
