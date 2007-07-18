import numpy as N

from recon.operations import Operation, verify_scanner_image
from recon.util import ifft, apply_phase_correction, linReg, shift, \
     checkerline, maskbyfit, unwrap_ref_volume

class UnbalPhaseCorrection (Operation):
    """
    Unbalanced Phase Correction attempts to reduce N/2 ghosting and other
    systematic phase errors by fitting referrence scan data to a system
    model. This can be run on Varian sequence EPI data acquired in 1 shot,
    multishot linear interleaved, or 2-shot centric sampling.
    """

    def run(self, image):
        # basic tasks here:
        # 1: data preparation
        # 2: phase unwrapping
        # 3: find mean phase lines (2 means or 4, depending on sequence)
        # 4: solve for linear coefficients
        # 5: create correction matrix from coefs
        # 6: apply correction to all image volumes
        #
        # *all linearly-sampled data can be treated in a generalized way by
        # paying attention to the interleave factor (self.xleave)
        # *centric sampled data, whose k-space trajectory had directions,
        # needs special treatment: basically the general case is handled in
        # separated parts
        if not verify_scanner_image(self, image):
            return -1
        if not hasattr(image, "ref_data"):
            self.log("No reference volume, quitting")
            return -1
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        self.volShape = image.shape[-3:]
        refVol = image.ref_data[0]        
        n_slice, n_pe, n_fe = self.refShape = refVol.shape
        # iscentric says whether kspace is multishot centric;
        # xleave is the factor to which kspace data has been interleaved
        # (in the case of multishot interleave)
        iscentric = image.sampstyle is "centric"
        self.xleave = iscentric and 1 or image.nseg
        self.alpha, self.beta = image.epi_trajectory()
        # get slice positions (in order) so we can throw out the ones
        # too close to the backplane of the headcoil
        acq_order = image.acq_order
        s_ind = N.concatenate([N.nonzero(acq_order==s)[0] for s in range(n_slice)])
        self.pss = N.take(image.slice_positions, s_ind)
        # want to fork the code based on sampling style
        if iscentric:
            theta = self.run_centric(ifft(refVol))
        else:
            theta = self.run_linear(ifft(refVol))

        phase = N.exp(-1.j*theta)
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = apply_phase_correction(image[:], phase)
        else:
            for dvol in image:
                dvol[:] = apply_phase_correction(dvol[:], phase)


    def run_linear(self, inv_ref):
        n_slice, n_pe, n_fe = self.refShape
        # conj order tells us how to make the unbalanced phase differences
        conj_order = N.arange(n_pe)
        shift(conj_order, 0, -self.xleave)
        inv_ref = N.conjugate(N.take(inv_ref, conj_order, axis=1)) * inv_ref
        # set up data indexing helpers, based on acquisition order.            
        # pos_order, neg_order define which rows in a slice are grouped
        # (remember not to count the lines contaminated by artifact!)
        pos_order = N.nonzero(self.alpha > 0)[0][self.xleave:]
        neg_order = N.nonzero(self.alpha < 0)[0][:-self.xleave]
        # comes back truncated to linear region:
        # from this point on, into the svd, work with truncated arrays
        # (still have "true" referrences from from self.refShape and self.lin1)
        #phs_vol = unwrap_ref_volume(inv_ref, self.lin1, self.lin2)
        phs_vol = unwrap_ref_volume(inv_ref, 0, n_fe)

        phs_mean, q1_mask = self.mean_and_mask(phs_vol[:,pos_order,:],
                                               phs_vol[:,neg_order,:])
            
        ### SOLVE FOR THE SYSTEM PARAMETERS FOR UNMASKED SLICES
        self.coefs = self.solve_phase(phs_mean, q1_mask)
        print self.coefs
        return self.correction_volume()

    def run_centric(self, inv_ref):
        n_slice, n_pe, n_fe = self.refShape
        # conj order tells us how to make the unbalanced phase differences
        conj_order = N.arange(n_pe)
        shift(conj_order[:n_pe/2],0,1)
        shift(conj_order[n_pe/2:],0,-1)
        inv_ref = N.conjugate(N.take(inv_ref, conj_order, axis=1)) * inv_ref
        # set up data indexing helpers, based on acquisition order.            
        # pos_order, neg_order define which rows in a slice are grouped
        pos_order = N.nonzero(self.alpha > 0)[0]
        neg_order = N.nonzero(self.alpha < 0)[0]
        # comes back truncated to linear region:
        # from this point on, into the svd, work with truncated arrays
        # (still have "true" referrences from from self.refShape, self.lin1, etc)
        phs_vol = unwrap_ref_volume(inv_ref, 0, n_fe)
        # for upper means, use
        phs_mean_upper, q1_mask_upper = \
                        self.mean_and_mask(phs_vol[:,pos_order[n_pe/4+1:],:],
                                           phs_vol[:,neg_order[n_pe/4:-1],:])

        phs_mean_lower, q1_mask_lower = \
                        self.mean_and_mask(phs_vol[:,pos_order[:n_pe/4-1],:],
                                           phs_vol[:,neg_order[1:n_pe/4],:])

        self.coefs = self.solve_phase(phs_mean_upper, q1_mask_upper)
        print self.coefs
        theta_upper = self.correction_volume()
        self.coefs = self.solve_phase(phs_mean_lower, q1_mask_lower)
        print self.coefs
        theta_lower = self.correction_volume()
        theta_lower[:,n_pe/2:,:] = theta_upper[:,n_pe/2:,:]
        return theta_lower

    def mean_and_mask(self, phs_evn, phs_odd):
        (n_slice, _, n_fe) = self.refShape
        phs_mean = N.empty((n_slice, 2, n_fe), N.float64)
        sigma = N.empty((n_slice, 2, n_fe), N.float64)
        phs_mean[:,0,:] = phs_evn.sum(axis=-2)/phs_evn.shape[-2]
        phs_mean[:,1,:] = phs_odd.sum(axis=-2)/phs_odd.shape[-2]
        sigma[:,0,:] = N.power(N.std(phs_evn, axis=-2), 2.0)
        sigma[:,1,:] = N.power(N.std(phs_odd, axis=-2), 2.0)
        q1_mask = N.ones((n_slice, 2, n_fe))
        bad_slices = (self.pss < -25.0)
        if bad_slices.any():
            last_good_slice = bad_slices.nonzero()[0][0]
        else: 
            last_good_slice = n_slice           
        q1_mask[last_good_slice:] = 0.0
        maskbyfit(phs_mean[:last_good_slice], sigma[:last_good_slice],
                      0.75, 2.0, q1_mask[:last_good_slice])
        # ditch any lines with less than 4 good pts
        for sl in q1_mask:
            npts = sl.sum(axis=-1)
            if npts[0] < 4:
                sl[0] = 0.0
            if npts[1] < 4:
                sl[1] = 0.0
        return phs_mean, q1_mask

    def solve_phase(self, phs, ptmask):
        """let V = (a1 a2 a3 a4 a5 a6)^T,
        we want to solve:
        phi(s,u,r) =  2[rA1 + sA3 + A5] - [rA2 + sA4 + A6] u-pos
        phi(s,u,r) = -2[rA1 + sA3 + A5] - [rA2 + sA4 + A6] u-neg
        with our overdetermined data.
        so P = [neg[0] pos[0] neg[1] pos[1] ... neg[S] pos[S]]^T
        for all selected slices, and similarly
        A = [-2r0 -r0 -2s0 -s0 -2 -1;  <--
             -2r1 -r1 -2s0 -s0 -2 -1;  <-- neg going u, s,r=0
             ...
             2r0 -r0 2s0 -s0 2 -1;     <--
             2r1 -r1 2s0 -s0 2 -1;     <-- pos going u, s,r=0
             ...
             -2r0 -r0 -2s1 -s1 -2 -1;  <-- neg going u, r=0,s=1
             ...
             2r0 -r0 2s1 -s1 2 -1;     <-- pos going u, r=0,s=1
             ...]
        Then with AV = P, solve V = inv(A)P
        """
        S,M,R = phs.shape
        A1, A2, A3, A4, A5, A6 = (0,1,2,3,4,5)
        # build the full matrix first, collapse the zero-rows afterwards
        nrows = N.product(phs.shape)
        A = N.zeros((nrows, 6), N.float64)
        P = N.reshape(phs, (nrows,))
        ptmask = N.reshape(ptmask, (nrows,))
        r_line = N.arange(R) - R/2
        s_line = N.arange(S)
        chk = N.outer(checkerline(M*S), N.ones(R))
        A[:,A1] = (2*chk*r_line).flatten()
        A[:,A2] = (-N.outer(N.ones(M*S), r_line)).flatten()
        A[:,A3] = 2*N.repeat(checkerline(M*S), R)*N.repeat(s_line, M*R)
        A[:,A4] = -N.repeat(s_line, M*R)
        A[:,A5] = (2*chk).flatten()
        A[:,A6] = -1.0

        nz = ptmask.nonzero()[0]
        A = A[nz]
        P = P[nz]

        [u,s,vt] = N.linalg.svd(A, full_matrices=0)
        V = N.dot(N.transpose(vt), N.dot(N.diag(1/s), N.dot(N.transpose(u),P)))
##         import pylab as pl
##         ptmask = N.reshape(ptmask, phs.shape)
##         for s in s_line:
##             r_ind_ev = N.nonzero(ptmask[s,0])[0]
##             r_ind_od = N.nonzero(ptmask[s,1])[0]
##             if r_ind_ev.any() and r_ind_od.any():
##                 pl.plot(r_ind_ev, N.take(phs[s,0], r_ind_ev), 'b')
##                 pl.plot(phs[s,0], 'b--')
##                 pl.plot(r_ind_od, N.take(phs[s,1], r_ind_od), 'r')
##                 pl.plot(phs[s,1], 'r--')
##                 rowpos = 2*r_line*V[A1] - r_line*V[A2] +\
##                          2*s*V[A3] - s*V[A4] + 2*V[A5] - V[A6]
##                 rowneg = -2*r_line*V[A1] - r_line*V[A2] + \
##                          -2*s*V[A3] - s*V[A4] - 2*V[A5] - V[A6]
##                 pl.plot(r_ind_ev, rowpos[r_ind_ev], 'b.')
##                 pl.plot(r_ind_od, rowneg[r_ind_od], 'r.')
##                 pl.title("slice %d"%s)
##                 pl.show()

        return V

    def correction_volume(self):
        """
        build the volume of phase correction lines with::
            theta(s,m,r) = m*[r*A2 + s*A4 + A6] + (-1)^m*[r*A1 + s*A3 + A5]

        or some scan-dependent variation (see doc)

        B is always 6 rows, the first two giving the r-dependencies, the last
        four being constants::
            B is (6 x n_fe) = [0:N*a1; 0:N*a2; a3; a4; a5; a6]
            A[s] represents the u and s dependencies: at each row, it multiplies
            values in the columns of B by the appropriate (m,s) relationship (thus
            A[s] changes for each value of s in the volume)
            A[s] is (n_pe x 6)
        """
        # the framework here is such that once zigzag[m] and m-line[m] are set
        # up correctly, theta[s] = A[s]*B ALWAYS!
        (S, M, R) = self.volShape
        (a1, a2, a3, a4, a5, a6) = self.coefs.tolist()
        A = N.empty((M, len(self.coefs)), N.float64)
        B = N.empty((len(self.coefs), R), N.float64)
        theta = N.empty(self.volShape, N.float64)
        
        # m_line & zigzag define how the correction changes per PE line
        # m_line is usually {-32,-31,...,30, 31}, but changes in multishot
        # zigzag[m] defines which rows goes negative or positive (is [-1,1])
        (zigzag, m_line) = (self.alpha, self.beta)

        # build B matrix, always stays the same
##         # ADD THIS PART TO CHANGE r->f(r)
##         from pylab import power
##         g = N.arange(R)-R/2
##         g = g/(1 + power(g/(R*.45), 6))
##         B[0] = g*a1[0]
##         B[1] = g*a2[0]
        B[0,:] = (N.arange(R)-R/2)*a1
        B[1,:] = (N.arange(R)-R/2)*a2
        B[2,:] = a3
        B[3,:] = a4
        B[4,:] = a5
        B[5,:] = a6
        
        # build A matrix, changes slightly as s varies
        A[:,0] = zigzag
        A[:,1] = m_line
        A[:,4] = zigzag
        A[:,5] = m_line
        for s in range(S):
            # these are the slice-dependent columns
            A[:,2] = s*zigzag
            A[:,3] = s*m_line
            theta[s] = N.dot(A,B)
            
        return theta
