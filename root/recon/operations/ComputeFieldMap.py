import numpy as np

from recon.punwrap import unwrap3D
from recon.imageio import clean_name, ReconImage
from recon.operations import Operation, Parameter, ChannelAwareOperation
from recon.operations.GaussianSmooth import GaussianSmooth

def build_3Dmask(vol, threshfactor):
    mask = np.ones(vol.shape, np.float32)
    p = np.sort(vol.flatten())
    npts = p.shape[0]
    t2 = p[int(round(.02*npts))]
    t98 = p[int(round(.98*npts))]
    thresh = threshfactor*(t98 - t2) + t2
    np.putmask(mask, vol<thresh, 0)
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
    @ChannelAwareOperation
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

        phase_map = image._subimage(np.zeros(diffshape, np.float32))
        bytemask = image._subimage(np.zeros(diffshape, np.float32))

        # Get phase difference between scan n+1 and scan n
        # Then find the mask and unwrapped phase, and compute
        # the field strength in terms of rad/s
        fwhm = max(image.isize, image.jsize) * 1.5
        for psub, msub in zip(phase_map, bytemask):
            del_te = asym_times[msub.num+1] - asym_times[msub.num]
            if self.haschans:
                for c in range(image.n_chan):
                    image.load_chan(c)
                    dphs = image[msub.num+1] * np.conjugate(image[msub.num])
                    msk = build_3Dmask(np.power(np.abs(dphs), 0.5),
                                       self.threshfactor)
                    msub[:] += msk
##                     psub[:] += (unwrap3D(np.angle(dphs)) * msk) #/ del_te
##                     psub[:] += gaussian_smooth(np.angle(dphs), 1.5, 3)
##                     smooth_phs = unwrap3D(gaussian_smooth(np.angle(dphs),
##                                                           sigma_y, sigma_x,
##                                                           gaussian_dims))*msk
                    smooth_phs = ReconImage(np.angle(dphs),
                                            isize=image.isize,
                                            jsize=image.jsize)
                    GaussianSmooth(fwhm=fwhm).run(smooth_phs)
                    psub[:] += smooth_phs[:]
                #psub[:] = np.where(msub[:], psub[:]/msub[:], 0)
                psub[:] /= image.n_chan
                msub[:] = np.where(msub[:], 1, 0)
                #psub[:] = (unwrap3D(psub[:]) * msub[:]) #/ del_te
                image.use_membuffer(0)
            else:
                dphs = image[msub.num+1] * np.conjugate(image[msub.num])
                msub[:] = build_3Dmask(np.power(np.abs(dphs), 0.5),
                                       self.threshfactor)
                #psub[:] = (unwrap3D(np.angle(dphs)) * msub[:]) #/ del_te
                psub[:] = np.angle(dphs)
                GaussianSmooth(fwhm=fwhm).run(psub)
                psub[:] = unwrap3D(psub[:])*msub[:]
                psub[:] /= del_te

        # for each diff vol, write a file with vol0 = fmap, vol1 = mask
        fmap_file = clean_name(self.fmap_file)[0]
        if phase_map.ndim > 3:
            for index in range(phase_map.tdim):
                catIm = phase_map.subImage(index).concatenate(
                    bytemask.subImage(index), newdim=True)
                catIm.writeImage(fmap_file+"-%d"%index,
                                 format_type="nifti-single")
        else:
            catIm = phase_map.concatenate(bytemask, newdim=True)
            catIm.writeImage(fmap_file, format_type="nifti-single")
