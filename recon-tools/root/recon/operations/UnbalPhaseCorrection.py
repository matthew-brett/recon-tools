import numpy as N

from recon.operations import Operation, verify_scanner_image, Parameter
from recon.util import ifft, apply_phase_correction, linReg, shift, \
     checkerline, maskbyfit, unwrap_ref_volume, reverse, normalize_angle

class UnbalPhaseCorrection (Operation):
    """
    Unbalanced Phase Correction attempts to reduce N/2 ghosting and other
    systematic phase errors by fitting referrence scan data to a system
    model. This can be run on Varian sequence EPI data acquired in 1 shot,
    multishot linear interleaved, or 2-shot centric sampling.
    """

    params = (Parameter(name="percentile", type="float", default=75.0,
                        description="""
    Indicates what percentage of "good quality" points to use in the solution.
    """),
              Parameter(name="shear_correct", type="bool", default=True,
                        description="""
    Switches shear correction on and off (takes values True and False)"""),
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
        # iscentric says whether kspace is multishot centric;
        # xleave is the factor to which kspace data has been interleaved
        # (in the case of multishot interleave)
        iscentric = image.sampstyle is "centric"
        self.xleave = iscentric and 1 or image.nseg
        self.alpha, self.beta = image.epi_trajectory()
        # get slice positions (in order) so we can throw out the ones
        # too close to the backplane of the headcoil
        self.good_slices = tag_backplane_slices(image)
        
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
        conj_order = N.array(range(self.xleave, n_pe) + range(self.xleave))
        inv_ref = N.conjugate(N.take(inv_ref, conj_order, axis=1)) * inv_ref
        # set up data indexing helpers, based on acquisition order.            
        # pos_order, neg_order define which rows in a slice are grouped
        # (remember not to count the lines contaminated by artifact!)
        pos_order = N.nonzero(self.alpha > 0)[0][self.xleave:]
        neg_order = N.nonzero(self.alpha < 0)[0][:-self.xleave]

        phs_vol = unwrap_ref_volume(inv_ref)
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
        phs_vol = unwrap_ref_volume(inv_ref)

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
        s_idx = self.good_slices
        phs_mean = N.empty((n_slice, 2, n_fe), N.float64)
        sigma = N.empty((n_slice, 2, n_fe), N.float64)
        phs_mean[:,0,:] = phs_evn.sum(axis=-2)/phs_evn.shape[-2]
        phs_mean[:,1,:] = phs_odd.sum(axis=-2)/phs_odd.shape[-2]
        #sigma[:,0,:] = N.power(N.std(phs_evn, axis=-2), 2.0)
        #sigma[:,1,:] = N.power(N.std(phs_odd, axis=-2), 2.0)
        q1_mask = N.zeros((n_slice, 2, n_fe))
        q1_mask[s_idx] = 1
        q1_mask[s_idx,0,:] = qual_map_mask(phs_evn[s_idx], self.percentile)
        q1_mask[s_idx,1,:] = qual_map_mask(phs_odd[s_idx], self.percentile)
        

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

        nz1 = ptmask.nonzero()[0]
        A = A[nz1]
        P = P[nz1]

        [u,s,vt] = N.linalg.svd(A, full_matrices=0)
        V = N.dot(vt.transpose(), N.dot(N.diag(1/s), N.dot(u.transpose(), P)))
        
##         import pylab as pl
##         ptmask = N.reshape(ptmask, phs.shape)
##         for s in s_line:
##             r_ind_ev = N.nonzero(ptmask[s,0])[0]
##             r_ind_od = N.nonzero(ptmask[s,1])[0]
##             if r_ind_ev.any() and r_ind_od.any():
##                 if self.shear_correct:
##                     rowpos = 2*r_line*V[A1] - r_line*V[A2] +\
##                              2*s*V[A3] - s*V[A4] + 2*V[A5] - V[A6]
##                     rowneg = -2*r_line*V[A1] - r_line*V[A2] + \
##                              -2*s*V[A3] - s*V[A4] - 2*V[A5] - V[A6]
##                 else:
##                     rowpos = 2*r_line*V[A1] + 2*s*V[A3] + 2*V[A5]
##                     rowneg = -(2*r_line*V[A1] + 2*s*V[A3] + 2*V[A5])
##                 pl.plot(rowpos, 'b.')
##                 pl.plot(rowneg, 'r.')
##                 pl.plot(phs[s,0], 'b--')
##                 pl.plot(r_ind_ev, phs[s,0,r_ind_ev], 'bo', alpha=0.25)

##                 pl.plot(phs[s,1], 'r--')
##                 pl.plot(r_ind_od, phs[s,1,r_ind_od], 'ro', alpha=0.25)
                
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
        if not self.shear_correct:
            a2 = a4 = a6 = 0.
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

def qual_map_mask(phs, pct):
    s,m,n = phs.shape
    qmask = N.zeros((s,n))    
    qual = N.zeros(( s-2,m-2,n-2 ), phs.dtype)

    hdiff = N.diff(phs[1:-1, 1:-1, :], n=2, axis=-1)
    vdiff = N.diff(phs[1:-1, :, 1:-1], n=2, axis=-2)
    udiff = N.diff(phs[:, 1:-1, 1:-1], n=2, axis=-3)
    sum_sqrs = N.power(hdiff, 2) + N.power(vdiff, 2) + N.power(udiff, 2)
    
    qual = N.power(sum_sqrs, 0.5).mean(axis=-2)
    qual = qual - qual.min()
    # set a major cutoff to throw out the real bad points
    cutoff = 0.2
    N.putmask(qmask[1:-1,1:-1], qual <= cutoff, 1)
    #################################################################
    
##     import pylab as P
##     import matplotlib.axes3d as p3
##     ax = P.axes()
##     ax.hold(True)
##     ax.imshow(qual)
##     ax.imshow(qmask[1:-1,1:-1], alpha=.6, cmap=P.cm.copper)
##     P.show()
##     Rx,Sx = P.meshgrid(N.arange(n-2), N.arange(s-2))
##     Rx2,Sx2 = P.meshgrid(N.arange(n), N.arange(s))
##     #cutoff = N.power(N.pi/8, 2)
##     bar = N.ones(qual.shape) * cutoff
##     fig = P.figure()
##     ax = p3.Axes3D(fig)
##     ax.hold(True)
##     ax.plot_wireframe(Rx,Sx,qual)
##     #ax.plot_wireframe(Rx,Sx,bar, colors=[1,0,0,1])
##     #P.show()
##     fig = P.figure()
##     ax = p3.Axes3D(fig)
##     ax.plot_wireframe(Rx2,Sx2, qmask*phs.mean(axis=-2))
##     P.show()

    #################################################################
    nz = qmask[1:-1,1:-1].flatten().nonzero()[0]
    x = N.sort(qual.flatten()[nz])
    npts = x.shape[0]
    cutoff = x[int(round(npts*pct/100.))]
    # prctile seems to be dangerous!!
    #cutoff = P.prctile(qual.flatten()[nz], pct)
    N.putmask(qmask[1:-1,1:-1], qual > cutoff, 0)
    #################################################################

##     P.hist(qual.flatten(), bins=50)
##     P.title("all qualities")
##     P.show()
##     P.hist(qual.flatten()[nz], bins=20)
##     P.title("after major cutoff")
##     P.show()    
##     nz = qmask[1:-1,1:-1].flatten().nonzero()[0]
##     P.hist(qual.flatten()[nz], bins=20)
##     P.title("after final %2.2fth percentile cutoff"%pct)
##     P.show()
##     ax = P.axes()
##     ax.hold(True)
##     ax.imshow(qual)
##     ax.imshow(qmask[1:-1,1:-1], alpha=.6, cmap=P.cm.copper)
##     P.show()
##     fig = P.figure()
##     ax = p3.Axes3D(fig)
##     ax.hold(True)
##     ax.plot_wireframe(Rx2, Sx2, qmask*phs.mean(axis=-2))
##     P.show()    

    #################################################################
    return qmask

def tag_backplane_slices(image):
    # 1st determine if the slice direction heads off towards the
    # backplane... this will be the case if:
    # theta is 0 +/- 10-deg (top slices) or 180 +/- 10-deg (bottom slices)
    # and if psi is 0 +/- 10-deg ( ||psi|| always < 90 ??? )
    # ignore phi, as this turns around the axis of the bore
    theta = image.theta - 360 * int(image.theta/180)
    psi = image.psi
    good_slices = N.arange(image.nslice)
    if psi >= -10 and psi <= 10:
        s_ind = N.concatenate([N.nonzero(image.acq_order==s)[0]
                               for s in range(image.nslice)])
        pss = N.take(image.slice_positions, s_ind)
        
        if (theta>=-10 and theta<=10) or (theta<=-170 and theta>=170):
            good_slices = (pss >= -25.0).nonzero()[0]
    return good_slices
