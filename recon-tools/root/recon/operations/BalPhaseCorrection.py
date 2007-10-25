"Applies a Balanced Phase Correction on data sets with two reference volumes"

import numpy as N
import os
from recon.operations import Operation, verify_scanner_image, Parameter
from recon.operations.UnbalPhaseCorrection import tag_backplane_slices
from recon.util import ifft, apply_phase_correction, unwrap_ref_volume, reverse

class BalPhaseCorrection (Operation):
    """
    Balanced Phase Correction attempts to reduce N/2 ghosting and other
    systematic phase errors by fitting referrence scan data to a system
    model. This can only be run on special balanced reference scan data.
    """
    params = (Parameter(name="percentile", type="float", default=90.0,
                        description="""
    Indicates what percentage of "good quality" points to use in the solution.
    """),
              Parameter(name="backplane_adj", type="bool", default=False,
                        description="""
    Try to keep data contaminated by backplane eddy currents out of solution.
    """),
              Parameter(name="fitmeans", type="bool", default=False,
                        description="""
    Fit evn/odd means rather than individual planes.
    """),
              )
    
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

        #phs_vol comes back shaped (n_slice, n_pe, lin2-lin1)
        phs_vol = unwrap_ref_volume(inv_ref)
        
        q1_mask = N.zeros((n_slice, n_pe, n_fe))

        # get slice positions (in order) so we can throw out the ones
        # too close to the backplane of the headcoil (or not ???)
        if self.backplane_adj:
            s_idx = tag_backplane_slices(image)
        else:
            s_idx = range(n_slice)
        q1_mask[s_idx] = 1.0
        q1_mask[s_idx,0::2,:] =  qual_map_mask(phs_vol[s_idx,0::2,:],
                                              self.percentile)
        q1_mask[s_idx,1::2,:] = qual_map_mask(phs_vol[s_idx,1::2,:],
                                              self.percentile)
        theta = N.empty(self.refShape, N.float64)
        s_line = N.arange(n_slice)
        r_line = N.arange(n_fe) - n_fe/2

        B1, B2, B3 = range(3)

        # planar solution
        nrows = n_slice * n_fe
        M = N.zeros((nrows, 3), N.float64)
        M[:,B1] = N.outer(N.ones(n_slice), r_line).flatten()
        M[:,B2] = N.repeat(s_line, n_fe)
        M[:,B3] = 1.
        
        A = N.empty((n_slice, 3), N.float64)
        B = N.empty((3, n_fe), N.float64)
        A[:,0] = 1.
        A[:,1] = s_line
        A[:,2] = 1.
        if not self.fitmeans:
            for m in range(n_pe):
                P = N.reshape(0.5*phs_vol[:,m,:], (nrows,))
                pt_mask = N.reshape(q1_mask[:,m,:], (nrows,))
                nz = pt_mask.nonzero()[0]
                Msub = M[nz]
                P = P[nz]
                [u,sv,vt] = N.linalg.svd(Msub, full_matrices=0)
                coefs = N.dot(vt.transpose(),
                              N.dot(N.diag(1/sv), N.dot(u.transpose(), P)))
                
                B[0,:] = coefs[B1]*r_line
                B[1,:] = coefs[B2]
                B[2,:] = coefs[B3]
                theta[:,m,:] = N.dot(A,B)
        else:
            for rows in ( 'evn', 'odd' ):
                if rows is 'evn':
                    slicing = ( slice(None), slice(0, n_pe, 2), slice(None) )
                else:
                    slicing = ( slice(None), slice(1, n_pe, 2), slice(None) )
                P = N.reshape(0.5*phs_vol[slicing].mean(axis=-2), (nrows,))
                pt_mask = q1_mask[slicing].prod(axis=-2)
                pt_mask.shape = (nrows,)
                nz = pt_mask.nonzero()[0]
                Msub = M[nz]
                P = P[nz]
                [u,sv,vt] = N.linalg.svd(Msub, full_matrices=0)
                coefs = N.dot(vt.transpose(),
                              N.dot(N.diag(1/sv), N.dot(u.transpose(), P)))
                B[0,:] = coefs[B1]*r_line
                B[1,:] = coefs[B2]
                B[2,:] = coefs[B3]
                theta[slicing] = N.dot(A,B)[:,None,:]
            
        phase = N.exp(-1.j*theta)
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = apply_phase_correction(image[:], phase)
        else:
            for dvol in image:
                dvol[:] = apply_phase_correction(dvol[:], phase)
        
def qual_map_mask(phs, pct):
    s,m,n = phs.shape
    qmask = N.ones((s,m,n))
    qmask[1:-1,1:-1,1:-1] = 0

    hdiff = N.diff(phs[1:-1, 1:-1, :], n=2, axis=-1)
    vdiff = N.diff(phs[1:-1, :, 1:-1], n=2, axis=-2)
    udiff = N.diff(phs[:, 1:-1, 1:-1], n=2, axis=-3)
    qual = N.power(hdiff, 2) + N.power(vdiff, 2) + N.power(udiff, 2)
    qual = N.power(qual, 0.5)
    qual = qual - qual.min()
    cutoff = 2.0
    N.putmask(qmask[1:-1,1:-1,1:-1], qual <= cutoff, 1)
    nz = qmask[1:-1,1:-1,1:-1].flatten().nonzero()[0]
    x = N.sort(qual.flatten()[nz])
    npts = x.shape[0]
    cutoff = x[int(round(npts*pct/100.))]    
    N.putmask(qmask[1:-1,1:-1,1:-1], qual > cutoff, 0)
    return qmask
