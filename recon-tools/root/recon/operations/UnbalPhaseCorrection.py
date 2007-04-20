import numpy as N

from recon.operations import Operation, Parameter, verify_scanner_image
from recon.util import ifft, apply_phase_correction, linReg, shift, \
     unwrap_ref_volume

class UnbalPhaseCorrection (Operation):
    """
    Unbalanced Phase Correction tries to determine six parameters which
    define the shape of reference scan phase lines via an SVD solution. Using
    these parameters, a volume of correction phases is built and applied. The
    influence of the parameters change slightly with different acquisition
    orders, as do the correction lines. Terminology used are "positive",
    "negative", referring to slope direction, and "upper", "lower", in the case
    of multishot-centric acquisition, where there are symmetric phase lines
    about the center row of k-space.
    """
    params = (
        Parameter(name="lin_radius", type="float", default=70.0,
            description="""
    Radius of the region of greatest linearity within the magnetic field,
    in mm (normally 70-80mm)"""),
        Parameter(name="thresh", type="float", default=0.1,
            description="""
    Mask points in the phase means whose standard deviation exceeds
    this number."""),
        )

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

        # [self.lin1:self.lin2] is the linear region of the gradient
        # self.lin_fe is the # of points in this region
        lin_pix = int(round(self.lin_radius/image.dFE))
        (self.lin1, self.lin2) = (lin_pix > n_fe/2) and (0,n_fe) or \
                                 ((n_fe/2-lin_pix), (n_fe/2+lin_pix))
        self.lin_fe = self.lin2-self.lin1
        self.FOV = image.dFE*n_fe
        # iscentric says whether kspace is multishot centric;
        # xleave is the factor to which kspace data has been interleaved
        # (in the case of multishot interleave)
        iscentric = image.sampstyle is "centric"
        self.xleave = iscentric and 1 or image.nseg
        self.alpha, self.beta = image.epi_trajectory()
        # want to fork the code based on sampling style
        if iscentric:
            theta = self.run_centric(ifft(refVol))
        else:
            theta = self.run_linear(ifft(refVol))
        if theta is None: 
            self.log("Could not find enough slices with sufficiently uniform\n"\
            "phase profiles. Try shortening the lin_radius parameter to\n"\
            "unwrap a more linear region, or bump up the thresh param.\n"\
            "Current FOV: %fmm, Current lin_radius: %fmm"%(self.FOV,
                                                           self.lin_radius))
            return -1
        
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = apply_phase_correction(image[:], -theta)
        else:
            for dvol in image:
                dvol[:] = apply_phase_correction(dvol[:], -theta)

    def run_linear(self, inv_ref):
        n_slice, n_pe, n_fe = self.refShape
        # conj order tells us how to make the unbalanced phase differences
        conj_order = N.arange(n_pe)
        shift(conj_order, 0, -self.xleave)
        inv_ref = N.conjugate(N.take(inv_ref, conj_order, axis=1)) * inv_ref
        # set up data indexing helpers, based on acquisition order.            
        # pos_order, neg_order define which rows in a slice are grouped
        pos_order = N.nonzero(self.alpha > 0)[0]
        neg_order = N.nonzero(self.alpha < 0)[0]
        # comes back truncated to linear region:
        # from this point on, into the svd, work with truncated arrays
        # (still have "true" referrences from from self.refShape and self.lin1)
        phs_vol = unwrap_ref_volume(inv_ref, self.lin1, self.lin2)
        phs_pos = N.empty((n_slice, self.lin_fe), N.float64)
        phs_neg = N.empty((n_slice, self.lin_fe), N.float64)
        s_mask = N.zeros(n_slice)
        r_mask = N.zeros((n_slice, self.lin_fe))
        res = N.zeros((n_slice,), N.float64)
        ### FIND THE MEANS FOR EACH SLICE
        for s in range(n_slice):
            # for pos_order, skip 1st 2 for xleave=2
            # for neg_order, skip last 2 for xleave=2
            phs_pos[s], mask_p, res_p = \
                        self.masked_avg(N.take(phs_vol[s],
                                               pos_order[self.xleave:],
                                               axis=0))
            phs_neg[s], mask_n, res_n = \
                        self.masked_avg(N.take(phs_vol[s],
                                               neg_order[:-self.xleave],
                                               axis=0))

            res[s] = res_p + res_n
            r_mask[s] = mask_p*mask_n

        # find 4 slices with smallest residual
        sres = N.sort(res)
        selected = [N.nonzero(res==c)[0] for c in sres[:4]]
        for c in selected:
            s_mask[c] = 1
            if(r_mask[c].sum() == 0):
                # if best slice's means are masked, simply return empty
                return
        ### SOLVE FOR THE SYSTEM PARAMETERS FOR UNMASKED SLICES
        self.coefs = self.solve_phase(phs_pos, phs_neg, r_mask, s_mask)
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
        phs_vol = unwrap_ref_volume(inv_ref, self.lin1, self.lin2)
        # set up some arrays for mean phases
        phs_pos_upper = N.empty((n_slice, self.lin_fe), N.float64)
        phs_pos_lower = N.empty((n_slice, self.lin_fe), N.float64)
        phs_neg_upper = N.empty((n_slice, self.lin_fe), N.float64)
        phs_neg_lower = N.empty((n_slice, self.lin_fe), N.float64)
        s_mask = N.zeros(n_slice)
        r_mask = N.zeros((n_slice, self.lin_fe))
        res = N.zeros((n_slice,), N.float64)
        for s in range(n_slice):
            # seems that for upper, skip 1st pos and last neg; 
            #            for lower, skip 1st pos and last neg
            phs_pos_upper[s], mask_pu, res_pu = \
                        self.masked_avg(N.take(phs_vol[s],
                                               pos_order[n_pe/4+1:],
                                               axis=0))
            phs_neg_upper[s], mask_nu, res_nu = \
                        self.masked_avg(N.take(phs_vol[s],
                                               neg_order[n_pe/4:n_pe/2-1],
                                               axis=0))
            phs_pos_lower[s], mask_pl, res_pl = \
                        self.masked_avg(N.take(phs_vol[s],
                                               pos_order[:n_pe/4-1],
                                               axis=0))
            phs_neg_lower[s], mask_nl, res_nl = \
                        self.masked_avg(N.take(phs_vol[s],
                                               neg_order[1:n_pe/4],
                                               axis=0))
                
            res[s] = res_pu + res_nu + res_pl + res_nl
            r_mask[s] = mask_pu*mask_nu*mask_pl*mask_nl

        
        # find 4 slices with smallest residual
        sres = N.sort(res)
        selected = [N.nonzero(res==c)[0] for c in sres[:4]]
        for c in selected:
            s_mask[c] = 1
            if(r_mask[c].sum() == 0):
                # if best slice's means are masked, simply return empty
                return
        ### SOLVE FOR THE SYSTEM PARAMETERS FOR UNMASKED SLICES
        # want to correct both segments "separately" (by splicing 2 thetas)
        v = self.solve_phase(phs_pos_upper, phs_neg_upper, r_mask, s_mask)
        self.coefs = v
        print self.coefs
        theta_upper = self.correction_volume()
        
        v = self.solve_phase(phs_pos_lower, phs_neg_lower, r_mask, s_mask)
        self.coefs = v
        print self.coefs        
        theta_lower = self.correction_volume()
        
        theta_vol = N.empty(theta_lower.shape, theta_lower.dtype)
        theta_vol[:,:n_pe/2,:] = theta_lower[:,:n_pe/2,:]
        theta_vol[:,n_pe/2:,:] = theta_upper[:,n_pe/2:,:]
        return theta_vol

    def masked_avg(self, S):
        """
        Take all pos or neg lines in a slice, return the mean of these lines
        along with a mask, where the mask is set to zero if the standard
        deviation exceeds a threshold (taken to mean noisy data), and also
        a sum of residuals from the linear fit of each line (taken as a measure
        of linearity)
        @param S is all pos or neg lines of a slice
        @return: E is the mean of these lines, mask is variance based mask, and
        sum(res) is the measure of linearity
        """
        
        nrow,npt = S.shape
        mask = N.ones((npt,))
        res = N.empty((nrow,), N.float64)

        E = N.add.reduce(S,axis=0)/float(S.shape[0])
        std = N.sqrt( ((S-E)**2/nrow).sum(axis=0) )
        N.putmask(mask, std>self.thresh, 0)

##         from pylab import show, plot, title
##         color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
##         for r in range(nrow):
##             plot(S[r], color[r%7])
##         plot(std, 'bo')
##         plot(E, 'go')
##         show()

        x_ax = N.nonzero(mask)[0]
        if mask.sum()>2:
            for r in range(nrow):
                (_, _, res[r]) = linReg(N.take(S[r], x_ax), X=x_ax)
        else: res = 1e10*N.ones(nrow)

        return E, mask, res.sum()

    def solve_phase(self, pos, neg, r_mask, s_mask):
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
        R = self.refShape[-1]
        A1, A2, A3, A4, A5, A6 = (0,1,2,3,4,5)
        # need 2*(sum-of-unmasked-points) rows for each selected slice
        n_rows = r_mask.sum(axis=1)
        s_ind = N.nonzero(s_mask)[0]
        A = N.empty((N.take(n_rows, s_ind).sum()*2, 6), N.float64)
        P = N.empty((N.take(n_rows, s_ind).sum()*2, 1), N.float64)
        row_start, row_end = 0, 0
        for s in s_ind:
            # alternate for pos and neg rows
            for b in [-1, 1]:
                row_start = row_end
                row_end = row_start + n_rows[s]
                r_ind = N.nonzero(r_mask[s])[0]
                if b > 0:
                    P[row_start:row_end,0] = N.take(pos[s], r_ind)
                else:
                    P[row_start:row_end,0] = N.take(neg[s], r_ind)
                # note these r_ind values are made relative to real fe points:
                A[row_start:row_end,A1] = b*2*(r_ind + self.lin1-R/2)
                A[row_start:row_end,A2] = -(r_ind + self.lin1-R/2)
                A[row_start:row_end,A3] = b*2*s
                A[row_start:row_end,A4] = -s
                A[row_start:row_end,A5] = b*2
                A[row_start:row_end,A6] = -1

        # take the SVD of A, and left-multiply its inverse to P              
        [u,s,vt] = N.linalg.svd(A, full_matrices=0)
        V = N.dot(N.transpose(vt), N.dot(N.diag(1/s), N.dot(N.transpose(u),P)))
##         import pylab as pl
##         for s in s_ind:
##             r_ind = N.nonzero(r_mask[s])[0]
##             pl.plot(r_ind, N.take(pos[s], r_ind), 'b')
##             pl.plot(r_ind, N.take(neg[s], r_ind), 'r')
##             rline = r_ind+self.lin1-R/2
##             rowpos = 2*rline*V[A1] - rline*V[A2] +\
##                      2*s*V[A3] - s*V[A4] + 2*V[A5] - V[A6]
##             rowneg = -2*rline*V[A1] - rline*V[A2] + \
##                      -2*s*V[A3] - s*V[A4] - 2*V[A5] - V[A6]
##             pl.plot(r_ind, rowpos, 'b.')
##             pl.plot(r_ind, rowneg, 'r.')
##             pl.show()

        return tuple(V) 

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
        (a1, a2, a3, a4, a5, a6) = self.coefs
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
        B[0,:] = (N.arange(R)-R/2)*a1[0]
        B[1,:] = (N.arange(R)-R/2)*a2[0]
        B[2,:] = a3[0]
        B[3,:] = a4[0]
        B[4,:] = a5[0]
        B[5,:] = a6[0]
        
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
