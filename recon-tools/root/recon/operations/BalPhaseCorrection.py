"Applies a Balanced Phase Correction on data sets with two reference volumes"

import numpy as N
import os
from recon.operations import Operation, verify_scanner_image
from recon.operations.ReorderSlices import ReorderSlices
from recon.util import ifft, apply_phase_correction, unwrap_ref_volume, \
     reverse, maskbyfit
from recon.imageio import readImage

class BalPhaseCorrection (Operation):
    """
    Balanced Phase Correction attempts to reduce N/2 ghosting and other
    systematic phase errors by fitting referrence scan data to a system
    model. This can only be run on special balanced reference scan data.
    """
    

    def run(self, image):
        
        if not verify_scanner_image(self, image):
            return -1
        
        if not hasattr(image, "ref_data") or image.ref_data.shape[0] < 2:
            self.log("Not enough reference volumes, quitting.")
            return -1

        self.volShape = image.shape[-3:]
        inv_ref0 = ifft(image.ref_data[0])
        inv_ref1 = ifft(reverse(image.ref_data[1], axis=-1))

        inv_ref = inv_ref0*N.conjugate(inv_ref1)
        
        n_slice, n_pe, n_fe = self.refShape = inv_ref0.shape

        # let's hardwire this currently??
        (self.lin1,self.lin2) = (0, n_fe)
        self.lin_fe = self.lin2-self.lin1
        self.alpha, self.beta = image.epi_trajectory()

        #phs_vol comes back shaped (n_slice, n_pe, lin2-lin1)
        phs_vol = unwrap_ref_volume(inv_ref, self.lin1, self.lin2)
        
        sigma = N.empty(phs_vol.shape, N.float64)
        #duplicate variance wrt to mu-ev/od over mu for convenience
        sigma[:,0::2,:] = N.power(N.std(phs_vol[:,0::2,:], axis=-2), 2.0)[:,None,:]
        sigma[:,1::2,:] = N.power(N.std(phs_vol[:,1::2,:], axis=-2), 2.0)[:,None,:]

        q1_mask = N.ones((n_slice, n_pe, self.lin_fe))

        # get slice positions (in order) so we can throw out the ones
        # too close to the backplane of the headcoil
        acq_order = image.acq_order
        s_ind = N.concatenate([N.nonzero(acq_order==s)[0] for s in range(n_slice)])
        pss = N.take(image.slice_positions, s_ind)
        bad_slices = (pss < -25.0)
        if bad_slices.any():
            last_good_slice = (pss < -25.0).nonzero()[0][0]
        else:
            last_good_slice = n_slice
        q1_mask[last_good_slice:] = 0.0
        maskbyfit(phs_vol[:last_good_slice],
                  sigma[:last_good_slice], 1.25, 1.25,
                  q1_mask[:last_good_slice])
        
        theta = N.empty(self.refShape, N.float64)
        s_line = N.arange(n_slice)
        r_line = N.arange(n_fe) - n_fe/2
        r_line_chop = N.arange(self.lin_fe) + self.lin1 - n_fe/2.

        B1, B2, B3 = range(3)

        # planar solution
        A = N.empty((n_slice, 3), N.float64)
        B = N.empty((3, n_fe), N.float64)
        A[:,0] = 1.
        A[:,1] = s_line
        A[:,2] = 1.
        for u in range(n_pe):
            coefs = solve_phase(0.5*phs_vol[:-1,u,:], q1_mask[:-1,u,:],
                                r_line_chop, s_line[:-1])
            B[0,:] = coefs[B1]*r_line
            B[1,:] = coefs[B2]
            B[2,:] = coefs[B3]
            theta[:,u,:] = N.dot(A,B)

        phase = N.exp(-1.j*theta)
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = apply_phase_correction(image[:], phase)
        else:
            for dvol in image:
                dvol[:] = apply_phase_correction(dvol[:], phase)
        
def solve_phase(pvol, surf_mask, r_line, s_line):
    # surface solution, pvol is (Nsl)x(Nro)
    # surf_mask is (Nsl)x(Nro)
    # r_line is 1xNro
    # s_line is 1xNsl
    # there is one row for each unmasked point in surf_mask
    (B1,B2,B3) = range(3)
    nrows = surf_mask.sum()
    A = N.empty((nrows, 3), N.float64)
    P = N.empty((nrows, 1), N.float64)
    row_start, row_end = 0, 0
    for s in range(surf_mask.shape[0]):
        
        unmasked = surf_mask[s].nonzero()[0]
        row_start = row_end
        row_end = unmasked.shape[0] + row_start
        P[row_start:row_end,0] = pvol[s,unmasked]
        A[row_start:row_end,B1] = r_line[unmasked]
        A[row_start:row_end,B2] = s_line[s]
        A[row_start:row_end,B3] = 1.
    
    [u,s,vt] = N.linalg.svd(A, full_matrices=0)
    V = N.dot(N.transpose(vt), N.dot(N.diag(1/s), N.dot(N.transpose(u),P)))

        
    return N.transpose(V)[0] 

