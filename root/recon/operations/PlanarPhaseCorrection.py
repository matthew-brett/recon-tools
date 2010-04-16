import numpy as np
from recon.operations import Operation, Parameter, ChannelIndependentOperation
from recon.scanners.siemens_utils import simple_unbal_phase_ramp
from recon import util
from recon.pmri.grappa_recon import grappa_sampling

class PlanarPhaseCorrection (Operation):
    """This is a fairly hacky operation to correct timing offsets in a
    possibly accelerated EPI acquisition, and ACS data when present.
    """
    params=(
      Parameter(name="fov_lim", type="tuple", default=None),
      Parameter(name="mask_noise", type="bool", default=True),
      )
    @ChannelIndependentOperation
    def run(self, image):
        arrays = (image.data,
                  image.acs_data if hasattr(image, 'acs_data') else None)
        samps = (image.n2_sampling, 
                 slice(None))
        refs = (image.ref_data,
                image.acs_ref_data if hasattr(image, 'acs_ref_data') else None)
        a,b = grappa_sampling(image.shape[-2], int(image.accel), image.n_acs)
        init_acs_dir = (-1.0) ** (a.tolist().index(b[0])) if len(b) else 1
        polarities = (1.0, init_acs_dir)
        nr = image.n_ramp
        nf = image.n_flat
        if image.pslabel == 'ep2d_bold_acs_test' or image.accel > 2:
            acs_xleave = int(image.accel)
        else:
            acs_xleave = 1
        
        xleaves = (1, int(acs_xleave))
        for arr, n2, ref_arr, r, x in zip(arrays, samps, refs, 
                                          polarities, xleaves):
            if arr is None:
                continue
            util.ifft1(arr, inplace=True, shift=True)
            # enforce 4D arrays
            arr.shape = (1,)*(4-len(arr.shape)) + arr.shape
            ref_arr.shape = (1,)*(4-len(ref_arr.shape)) + ref_arr.shape
            sub_samp_slicing = [slice(None)]*len(arr.shape)
            sub_samp_slicing[-2] = n2
            Q2, Q1 = arr[sub_samp_slicing].shape[-2:]
            q1_ax = np.linspace(-Q1/2., Q1/2., Q1, endpoint=False)
            q2_pattern = r*np.power(-1.0, np.arange(Q2)/x)
            for v in xrange(arr.shape[0]):
                sub_samp_slicing[0] = v
                for sl in range(arr.shape[1]):
                    sub_samp_slicing[1] = sl
                    m = simple_unbal_phase_ramp(ref_arr[v,sl].copy(),
                                                nr, nf, image.pref_polarity,
                                                fov_lim=self.fov_lim,
                                                mask_noise=self.mask_noise)
                    soln_pln = (m * q1_ax[None,:]) * q2_pattern[:,None]
                    phs = np.exp(-1j*soln_pln)
                    arr[sub_samp_slicing] *= phs
            # correct for flat dimensions
            arr.shape = tuple([d for d in arr.shape if d > 1])
            util.fft1(arr, inplace=True, shift=True)
        #image.ref_data = refs[0]
        #image.acs_ref_data = refs[1]
