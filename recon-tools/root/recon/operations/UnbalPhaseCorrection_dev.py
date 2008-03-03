import numpy as N

from recon.operations import Operation, verify_scanner_image, Parameter
from recon.operations.ComputeFieldMap import build_3Dmask
from recon.util import ifft, apply_phase_correction, linReg, shift, \
     checkerline, maskbyfit, unwrap_ref_volume, reverse, normalize_angle

class UnbalPhaseCorrection_dev (Operation):
    """
    Unbalanced Phase Correction attempts to reduce N/2 ghosting and other
    systematic phase errors by fitting referrence scan data to a system
    model. This can be run on Varian sequence EPI data acquired in 1 shot,
    multishot linear interleaved, or 2-shot centric sampling.
    """

    params = (Parameter(name="percentile", type="float", default=25.0,
                        description="""
    Indicates what percentage of "good quality" points to use in the solution.
    """),
              Parameter(name="svdsize", type="int", default=3, description=""),
              Parameter(name="fullcorr", type="bool", default=False)
              )


    def run(self, image):
        # basic tasks here:
        # 1: data preparation
        # 2: phase unwrapping
        # 3: find mean phase diff lines (2 means or 4, depending on sequence)
        # 4: solve for linear coefficients
        # 5: create correction matrix from coefs
        # 6: apply correction to all image volumes
        #
        # * all linearly-sampled data can be treated in a generalized way by
        #   paying attention to the interleave factor (self.xleave) and
        #   the sampling trajectory+timing of each row (image.epi_trajectory)
        #
        # * centric sampled data, whose k-space trajectory had opposite
        #   directions, needs special treatment: basically the general case
        #   is handled in separated parts

##         if not verify_scanner_image(self, image):
##             return -1
        if not hasattr(image, "ref_data"):
            self.log("No reference volume, quitting")
            return -1
        if len(image.ref_data.shape) > 3 and image.ref_data.shape[-4] > 1:
            self.log("Could be performing Balanced Phase Correction!")

        self.volShape = image.shape[-3:]
        refVol = image.ref_data[0]        
        n_slice, n_ref_rows, n_fe = self.refShape = refVol.shape
        # iscentric says whether kspace is multishot centric;
        # xleave is the factor to which kspace data has been interleaved
        # (in the case of multishot interleave)
        iscentric = image.sampstyle is "centric"
        self.xleave = iscentric and 1 or image.nseg
        self.alpha, self.beta, _ = image.epi_trajectory()
        # get slice positions (in order) so we can throw out the ones
        # too close to the backplane of the headcoil
        #self.good_slices = tag_backplane_slices(image)
        self.good_slices = range(n_slice)
        
        # want to fork the code based on sampling style
        if iscentric:
            theta = self.run_centric(image)
        else:
            theta = self.run_linear(image)

        if not self.fullcorr:
            phase = N.exp(1.j*theta)
            from recon.tools import Recon
            if Recon._FAST_ARRAY:
                image[:] = apply_phase_correction(image[:], phase)
            else:
                for dvol in image:
                    dvol[:] = apply_phase_correction(dvol[:], phase)
        else:
            # theta is the [q3, n2, n1, n1p] kernel
            ndim = image.ndim
            sn_shape = image.shape[-2:]
            if ndim > 3: sn_shape = (image.shape[0],) + sn_shape
            sn = N.empty(sn_shape, image[:].dtype)
            snslice = [ slice(None) ] * len(sn_shape)
            dslice = [ slice(None) ] * ndim
            # this is for shape [nvol, Q3, N2, N1, N1P]
            dslice1 = [ slice(None) ] * (ndim+1)
            dslice1[-2] = None # this is a placeholder for N1
            for q3 in range(self.volShape[-3]):
                print "correcting slice", q3
                dslice[-3] = q3
                dslice1[-4] = q3
                for n2 in range(self.volShape[-2]):
                    dslice1[-3] = n2
                    snslice[-2] = n2
                    sn[snslice] = (N.linalg.inv(theta[q3,n2]) * \
                                   image[dslice1]).sum(axis=-1)
                image[dslice] = sn[:]

    def run_linear(self, image):
        n_slice, n_ref_rows, n_fe = self.refShape

        N1 = image.shape[-1]
        Tr = hasattr(image, 'T_ramp') and float(image.T_ramp) or 1
        T0 = hasattr(image, 'T0') and float(image.T0) or 1
        Tf = hasattr(image, 'T_flat') and float(image.T_flat) or \
             image.delT * (N1-1)
        As = (Tr + Tf - (T0**2)/Tr)

        nr = int( (N1-1)*(Tr**2 - T0**2)/(2*Tr*As) )
        nf = int( (N1-1)*(2*Tf*Tr + Tr**2 - T0**2)/(2*Tr*As) )
        print nr, nf
        image.ref_data[...,:nr] = 0.
        image.ref_data[...,nf+1:] = 0.
        if (nr > 1) and (nf < N1-1):
            rsmooth(image.ref_data, nr, nf)
        
        n_conj_rows = n_ref_rows-self.xleave
        # form the S[u]S*[u+1] array:
        inv_ref = ifft(image.ref_data[0])
        inv_ref = inv_ref[:,:-self.xleave,:] * \
                  N.conjugate(inv_ref[:,self.xleave:,:])

        # Adjust the percentile parameter to reflect the percentage of
        # points that actually have data (not the pctage of all points).
        # Do this by determining the fraction of points that pass an
        # intensity threshold masking step.
        ir_mask = build_3Dmask(N.abs(inv_ref), 0.1)
        self.percentile *= ir_mask.sum()/(n_conj_rows*n_slice*n_fe)

        # partition the phase data based on acquisition order:
        # pos_order, neg_order define which rows in a slice are grouped
        # (remember not to count the lines contaminated by artifact!)
        pos_order = (self.alpha[:n_conj_rows] > 0).nonzero()[0]
        neg_order = (self.alpha[:n_conj_rows] < 0).nonzero()[0]

        # in Varian scans, the phase of the 0th product seems to be
        # contaminated.. so throw it out if there is at least one more
        # even-odd product
        # case < 3 ref rows: can't solve problem
        # case 3 ref rows: p0 from (0,1), n0 from (1,2)
        # case >=4 ref rows: p0 from (2,3), n0 from (1,2) (can kick line 0)
        # if the amount of data can support it, throw out p0
        if len(pos_order) > 1:
            pos_order = pos_order[1:]
        phs_vol = unwrap_ref_volume(inv_ref)
        phs_mean, q1_mask = mean_and_mask(phs_vol[:,pos_order,:],
                                          phs_vol[:,neg_order,:],
                                          self.percentile, self.good_slices)
        ### SOLVE FOR THE SYSTEM PARAMETERS
        if self.svdsize == 3:
            coefs = solve_phase_3d(phs_mean, q1_mask)
            print coefs
            if self.fullcorr:
                phs = phsfunc(image, coefs[0], coefs[1], coefs[2])
                return kern_full(phs)
            else:
                return correction_volume_3d(self.volShape, self.alpha, *coefs)
        else:
            coefs = solve_phase_6d(phs_mean, q1_mask)
            print coefs
            return correction_volume_6d(self.volShape, self.alpha,
                                        self.beta, *coefs)
        

    def run_centric(self, image):
        # centric sampling for epidw goes [0,..,31] then [-1,..,-32]
        # in index terms this is [32,33,..,63] + [31,30,..,0]

        # solving for angle(S[u]S*[u+1]) is equal to the basic problem for u>=0
        # for u<0:
        # angle(S[u]S*[u+1]) =   2[sign-flip-terms]*(-1)^(u+1) + [shear-terms]
        #                    = -(2[sign-flip-terms]*(-1)^u     - [shear-terms])
        # so by flipping the sign on the phs means data, we can solve for the
        # sign-flipping (raster) terms and the DC offset terms with the same
        # equations.
        
        n_ref_rows = self.refShape[-2]
        n_vol_rows = self.volShape[-2]
        # this is S[u]S*[u+1].. now with n_ref_rows-1 rows
        inv_ref = ifft(image.ref_data[0])
        inv_ref = inv_ref[:,:-1,:] * N.conjugate(inv_ref[:,1:,:])

        # Adjust the percentile parameter to reflect the percentage of
        # points that actually have data (not the pctage of all points).
        # Do this by determining the fraction of points that pass an
        # intensity threshold masking step.
        ir_mask = build_3Dmask(N.abs(inv_ref), 0.1)
        self.percentile *= ir_mask.sum()/(n_conj_rows*(n_slice*n_fe))
        
        # in the lower segment, do NOT grab the n_ref_rows/2-th line..
        # its product spans the two segments
        cnj_upper = inv_ref[:,n_ref_rows/2:,:].copy()
        cnj_lower = inv_ref[:,:n_ref_rows/2-1,:].copy()

        phs_evn_upper = unwrap_ref_volume(cnj_upper[:,0::2,:])
        phs_odd_upper = unwrap_ref_volume(cnj_upper[:,1::2,:])
        # 0th phase diff on the upper trajectory is contaminated by eddy curr,
        # throw it out if possible:
        if phs_evn_upper.shape[-2] > 1:
            phs_evn_upper = phs_evn_upper[:,1:,:]
        phs_evn_lower = unwrap_ref_volume(cnj_lower[:,0::2,:])
        phs_odd_lower = unwrap_ref_volume(cnj_lower[:,1::2,:])
        # 0th phase diff on downward trajectory (== S[u]S*[u+1] for u=-30)
        # is contaminated too
        if phs_evn_lower.shape[-2] > 1:
            phs_evn_lower = phs_evn_lower[:,:-1,:]
        
        phs_mean_upper, q1_mask_upper = \
                        mean_and_mask(phs_evn_upper, phs_odd_upper,
                                      self.percentile, self.good_slices)
        phs_mean_lower, q1_mask_lower = \
                        mean_and_mask(phs_evn_lower, phs_odd_lower,
                                      self.percentile, self.good_slices)
        if self.svdsize == 3:
            # for upper (u>=0), solve normal SVD
            coefs = solve_phase_3d(phs_mean_upper, q1_mask_upper)
            print coefs
            theta_upper = correction_volume_3d(self.volShape,self.alpha,*coefs)
            # for lower (u < 0), solve with negative data
            coefs = solve_phase_3d(-phs_mean_lower, q1_mask_lower)
            print coefs
            theta_lower = correction_volume_3d(self.volShape,self.alpha,*coefs)
            theta_lower[:,n_vol_rows/2:,:] = theta_upper[:,n_vol_rows/2:,:]
            return theta_lower
        else:
            # for upper (u>=0), solve normal SVD
            coefs = solve_phase_6d(phs_mean_upper, q1_mask_upper)
            print coefs
            theta_upper = correction_volume_6d(self.volShape, self.alpha,
                                               self.beta, *coefs)
            # for lower (u < 0), solve with negative data
            coefs = solve_phase_6d(-phs_mean_lower, q1_mask_lower)
            print coefs
            theta_lower = correction_volume_6d(self.volShape, self.alpha,
                                               self.beta, *coefs)
            theta_lower[:,n_vol_rows/2:,:] = theta_upper[:,n_vol_rows/2:,:]
            return theta_lower

def mean_and_mask(phs_evn, phs_odd, percentile, s_idxs):
    n_slice = phs_evn.shape[0]
    n_fe = phs_evn.shape[-1]
    # some checks to ensure generic performance
    if len(phs_evn.shape) < 3: phs_evn.shape = (n_slice, 1, n_fe)
    if len(phs_odd.shape) < 3: phs_odd.shape = (n_slice, 1, n_fe)
    phs_mean = N.empty((n_slice, 2, n_fe), N.float64)
    sigma = N.empty((n_slice, 2, n_fe), N.float64)
    phs_mean[:,0,:] = phs_evn.sum(axis=-2)/phs_evn.shape[-2]
    phs_mean[:,1,:] = phs_odd.sum(axis=-2)/phs_odd.shape[-2]
    q1_mask = N.zeros((n_slice, 2, n_fe))
    q1_mask[s_idxs] = 1
    q1_mask[s_idxs,0,:] = qual_map_mask2(phs_evn[s_idxs], percentile)
    q1_mask[s_idxs,1,:] = qual_map_mask2(phs_odd[s_idxs], percentile)
    
    return phs_mean, q1_mask


def solve_phase_3d(phs, ptmask):
    """
    let V = (a1 a3 a0)^T,
    we want to solve:
    phi(q3,m2,q1) = (-1)^(m2) * 2[q1*A1 + q3*A3 + A0]
    with our overdetermined data.
    so P = [neg[q3=0] pos[q3=0] neg[q3=1] pos[q3=1] ... neg[S] pos[S]]^T
    for all selected slices, and similarly
    A = [-2q1_0 -2q3_0 -2;
         -2q1_1 -2q3_0 -2;  <-- phi(q3=0,m2=odd; q1)  
             ...
          2q1_0  2q3_0  2;
          2q1_1  2q3_0  2;  <-- phi(q3=0,m2=evn; q1)
             ...
         -2q1_0 -2q3_1 -2;
         -2q1_1 -2q3_1 -2;  <-- phi(q3=1,m2=odd; q1)
             ...
          2q1_1  2q3_1  2;  <-- phi(q3=1,m2=evn; q1)
             ...          ]

    Then with AV = P, solve V = inv(A)P
    """
    Q3,M2,Q1 = phs.shape
    A1, A3, A0 = (0,1,2)
    # build the full matrix first, collapse the zero-rows afterwards
    nrows = N.product(phs.shape)
    A = N.zeros((nrows, 3), N.float64)
    P = phs.copy()
    P.shape = (nrows,)
    ptmask = N.reshape(ptmask, (nrows,))
    q1_line = N.linspace(-Q1/2., Q1/2., Q1, endpoint=False)
    q3_line = N.linspace(0., Q3, Q3, endpoint=False)
    m2sign = N.outer(checkerline(M2*Q3), N.ones(Q1))
    A[:,A1] = (2*m2sign*q1_line).flatten()
    A[:,A3] = 2*N.repeat(checkerline(M2*Q3), Q1)*N.repeat(q3_line, M2*Q1)
    A[:,A0] = (2*m2sign).flatten()
    
    nz1 = ptmask.nonzero()[0]
    A = A[nz1]
    P = P[nz1]
    
    [u,s,vt] = N.linalg.svd(A, full_matrices=0)
    V = N.dot(vt.transpose(), N.dot(N.diag(1/s), N.dot(u.transpose(), P)))
        
##     import pylab as pl
##     ptmask = N.reshape(ptmask, phs.shape)
##     for s in q3_line:
##         q1_ind_ev = N.nonzero(ptmask[s,0])[0]
##         q1_ind_od = N.nonzero(ptmask[s,1])[0]
##         if q1_ind_ev.any() and q1_ind_od.any():
##             rowpos = 2*q1_line*V[A1] + 2*s*V[A3] + 2*V[A0]
##             rowneg = -(2*q1_line*V[A1] + 2*s*V[A3] + 2*V[A0])
##             pl.plot(rowpos, 'b.')
##             pl.plot(rowneg, 'r.')
##             pl.plot(phs[s,0], 'b--')
##             pl.plot(q1_ind_ev, phs[s,0,q1_ind_ev], 'bo', alpha=0.25)
            
##             pl.plot(phs[s,1], 'r--')
##             pl.plot(q1_ind_od, phs[s,1,q1_ind_od], 'ro', alpha=0.25)
            
##             pl.title("slice %d"%s)
##             pl.show()
            
    return V

def solve_phase_6d(phs, ptmask):
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
    Q3,M2,Q1 = phs.shape
    A1, A2, A3, A4, A5, A6 = (0,1,2,3,4,5)
    # build the full matrix first, collapse the zero-rows afterwards
    nrows = N.product(phs.shape)
    A = N.zeros((nrows, 6), N.float64)
    P = N.reshape(phs, (nrows,))
    ptmask = N.reshape(ptmask, (nrows,))
    q1_line = N.arange(Q1) - Q1/2
    q3_line = N.arange(Q3)
    m2sign = N.outer(checkerline(M2*Q3), N.ones(Q1))
    A[:,A1] = (2*m2sign*q1_line).flatten()
    A[:,A2] = (-N.outer(N.ones(M2*Q3), q1_line)).flatten()
    A[:,A3] = 2*N.repeat(checkerline(M2*Q3), Q1)*N.repeat(q3_line, M2*Q1)
    A[:,A4] = -N.repeat(q3_line, M2*Q1)
    A[:,A5] = (2*m2sign).flatten()
    A[:,A6] = -1.0
    
    nz1 = ptmask.nonzero()[0]
    A = A[nz1]
    P = P[nz1]

    [u,s,vt] = N.linalg.svd(A, full_matrices=0)
    V = N.dot(vt.transpose(), N.dot(N.diag(1/s), N.dot(u.transpose(), P)))
        
##     import pylab as pl
##     ptmask = N.reshape(ptmask, phs.shape)
##     for s in q3_line:
##         r_ind_ev = N.nonzero(ptmask[s,0])[0]
##         r_ind_od = N.nonzero(ptmask[s,1])[0]
##         if r_ind_ev.any() and r_ind_od.any():
## ##             if self.shear_correct:
## ##                 rowpos = 2*q1_line*V[A1] - q1_line*V[A2] +\
## ##                          2*s*V[A3] - s*V[A4] + 2*V[A5] - V[A6]
## ##                 rowneg = -2*q1_line*V[A1] - q1_line*V[A2] + \
## ##                          -2*s*V[A3] - s*V[A4] - 2*V[A5] - V[A6]
## ##             else:
##             rowpos = 2*q1_line*V[A1] + 2*s*V[A3] + 2*V[A5]
##             rowneg = -(2*q1_line*V[A1] + 2*s*V[A3] + 2*V[A5])
##             pl.plot(rowpos, 'b.')
##             pl.plot(rowneg, 'r.')
##             pl.plot(phs[s,0], 'b--')
##             pl.plot(r_ind_ev, phs[s,0,r_ind_ev], 'bo', alpha=0.25)
            
##             pl.plot(phs[s,1], 'r--')
##             pl.plot(r_ind_od, phs[s,1,r_ind_od], 'ro', alpha=0.25)
            
##             pl.title("slice %d"%s)
##             pl.show()

    return V


def correction_volume_3d(shape, m2sign, a1, a3, a0):
    """
    build the volume of phase correction lines with::
    theta(q3,m2,q1) = (-1)^m2*[q1*A1 + q3*A3 + A0]
    
    or some scan-dependent variation (see doc)
    """

    (Q3, M2, Q1) = shape
    Qmesh = N.indices((Q3,Q1), dtype=N.float64)
    soln_plane = a3*Qmesh[0] + a1*(Qmesh[1] - Q1/2.) + a0
    soln_plane = soln_plane[:,None,:] * m2sign[None,:,None]
    return soln_plane

def correction_volume_6d(shape, m2sign, m2idx, a1, a2, a3, a4, a5, a6):
    """
    build the volume of phase correction lines with::
    theta(q3,m2,q1) = (-1)^m2*[q1*A1 + q3*A3 + A5] + m2*[q1*A2 + q3*A4 + A6]
    
    or some scan-dependent variation (see doc)
    """

    (Q3, M2, Q1) = shape
    Qmesh = N.indices((Q3,Q1), dtype=N.float64)
    p1 = a3*Qmesh[0] + a1*(Qmesh[1] - Q1/2.) + a5
    p2 = a4*Qmesh[0] + a2*(Qmesh[1] - Q1/2.) + a6
    soln_plane = (p1[:,None,:] * m2sign[None,:,None]) + \
                 (p2[:,None,:] * m2idx[None,:,None])
    return soln_plane

def qual_map_mask(phs, pct):
    s,m,n = phs.shape
    qmask = N.zeros((s,n))    
    if m>2:
        xdiff = N.diff(phs[1:-1, 1:-1, :], n=2, axis=-1)
        ydiff = N.diff(phs[1:-1, :, 1:-1], n=2, axis=-2)
        zdiff = N.diff(phs[:, 1:-1, 1:-1], n=2, axis=-3)
        sum_sqrs = N.power(xdiff, 2) + N.power(ydiff, 2) + N.power(zdiff, 2)
    else:
        xdiff = N.diff(phs[1:-1,:,:], n=2, axis=-1)
        zdiff = N.diff(phs[:,:,1:-1], n=2, axis=-3)
        sum_sqrs = N.power(xdiff, 2) + N.power(zdiff, 2)
    
    qual = N.power(sum_sqrs, 0.5).mean(axis=-2)
    qual = qual - qual.min()
    # set a major cutoff to throw out the real bad points
    # this might be about 10% of the stdev???
    #cutoff = 0.2
    cutoff = 0.1 * qual.std()
    #print cutoff
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
##     ax.hold(True)
##     #ax.plot_surface(Rx2,Sx2, N.zeros(qmask.shape))
##     ax.plot_wireframe(Rx2,Sx2, phs.mean(axis=-2), colors=(1,0,0,1))
##     P.show()

    #################################################################
    nz = qmask[1:-1,1:-1].flatten().nonzero()[0]
    x = N.sort(qual.flatten()[nz])
    npts = x.shape[0]
    cutoff = x[int(round(npts*pct/100.))]
    N.putmask(qmask[1:-1,1:-1], qual > cutoff, 0)
    #################################################################

##     #P.hist(qual.flatten(), bins=50)
##     #P.title("all qualities")
##     #P.show()
##     #P.hist(qual.flatten()[nz], bins=20)
##     #P.title("after major cutoff")
##     #P.show()    
##     #nz = qmask[1:-1,1:-1].flatten().nonzero()[0]
##     #P.hist(qual.flatten()[nz], bins=20)
##     #P.title("after final %2.2fth percentile cutoff"%pct)
##     #P.show()
##     ax = P.axes()
##     ax.hold(True)
##     ax.imshow(qual)
##     ax.imshow(qmask[1:-1,1:-1], alpha=.6, cmap=P.cm.copper)
##     P.show()
##     #fig = P.figure()
##     #ax = p3.Axes3D(fig)
##     #ax.hold(True)
##     #ax.plot_wireframe(Rx2, Sx2, qmask*phs.mean(axis=-2))
##     P.imshow(qmask*phs.mean(axis=-2))
##     P.colorbar()
##     P.show()
##     fig = P.figure()
##     ax = p3.Axes3D(fig)
##     ax.hold(True)
##     ax.plot_surface(Rx2,Sx2, N.zeros(qmask.shape))
##     ax.plot_wireframe(Rx2,Sx2, qmask*phs.mean(axis=-2), colors=(1,0,0,1))
##     P.show()

    #################################################################
    return qmask

def qual_map_mask2(phs, pct):
    # in this one, pct specifies the absolute percentage of points to use
    s,m,n = phs.shape
    qmask = N.zeros((s,n))    
    if m>2:
        xdiff = N.diff(phs[1:-1, 1:-1, :], n=2, axis=-1)
        ydiff = N.diff(phs[1:-1, :, 1:-1], n=2, axis=-2)
        zdiff = N.diff(phs[:, 1:-1, 1:-1], n=2, axis=-3)
        sum_sqrs = N.power(xdiff, 2) + N.power(ydiff, 2) + N.power(zdiff, 2)
    else:
        xdiff = N.diff(phs[1:-1,:,:], n=2, axis=-1)
        zdiff = N.diff(phs[:,:,1:-1], n=2, axis=-3)
        sum_sqrs = N.power(xdiff, 2) + N.power(zdiff, 2)
    
    qual = N.power(sum_sqrs, 0.5).mean(axis=-2)
    qual_s = qual.flatten().copy()
    qual_s.sort()
    npts = (s-2)*(n-2)
    cutoff = qual_s[int(round(npts*pct/100.))]
    N.putmask(qmask[1:-1,1:-1], qual <= cutoff, 1)
    #################################################################
    
##     import pylab as P
##     import matplotlib.axes3d as p3
##     Rx,Sx = P.meshgrid(N.arange(n-2), N.arange(s-2))
##     Rx2,Sx2 = P.meshgrid(N.arange(n), N.arange(s))
##     ax = P.axes()
##     ax.hold(True)
##     ax.imshow(qual)
##     ax.imshow(qmask[1:-1,1:-1], alpha=.6, cmap=P.cm.copper)
##     P.show()
##     #fig = P.figure()
##     #ax = p3.Axes3D(fig)
##     #ax.hold(True)
##     #ax.plot_wireframe(Rx2, Sx2, qmask*phs.mean(axis=-2))
##     P.imshow(qmask*phs.mean(axis=-2))
##     P.colorbar()
##     P.show()
##     fig = P.figure()
##     ax = p3.Axes3D(fig)
##     ax.hold(True)
##     ax.plot_surface(Rx2,Sx2, N.zeros(qmask.shape))
##     ax.plot_wireframe(Rx2,Sx2, qmask*phs.mean(axis=-2), colors=(1,0,0,1))
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
    good_slices = N.arange(image.n_slice)
    if psi >= -10 and psi <= 10:
        s_ind = N.concatenate([N.nonzero(image.acq_order==s)[0]
                               for s in range(image.n_slice)])
        pss = N.take(image.slice_positions, s_ind)
        
        if (theta>=-10 and theta<=10) or (theta<=-170 and theta>=170):
            good_slices = (pss >= -25.0).nonzero()[0]
    return good_slices


def rsmooth(f, nr, nf):
    s_size = 3
    avgr = N.ones(s_size, N.float64)/s_size
    fs = f.copy()
    for x in range(nr-s_size/2,nr+1):
        fs[...,x] = (avgr*f[...,x-s_size/2:x+s_size/2+1]).sum(axis=-1)
    for x in range(nf+1,nf+1+s_size/2):
        fs[...,x] = (avgr*f[...,x-s_size/2:x+s_size/2+1]).sum(axis=-1)
    return fs


def phsfunc(image, a1, a3, a0):

    Q3, Q2, Q1 = image.shape[-3:]
    N2, N1 = image.shape[-2:]

    T0 = image.T0
    Tr = image.Tr
    Tf = image.Tf
    delT = (Tr + Tf - (T0**2)/Tr)/(N1 - 1.0)

    # this is (-1)^(n2)
    n2sign = checkerline(N2)
    
    # fn is shaped (N2, N1)
    fn = N.outer(0.5*(1-n2sign)*(N1-1), N.ones(N1)) + N.outer(n2sign, N.arange(N1))
    
    # soln_plane is shaped (Q3, Q1)
    Qplane = N.indices((Q3, Q1), dtype=N.float64)
    soln_plane = a3*Qplane[0] + a1*(Qplane[1]-Q1/2) + a0
    
    nr = int(N.floor( (Tr/2. - T0**2/(2.*Tr))/delT ))
    nf = int(N.floor( (Tr/2. + Tf - T0**2/(2.*Tr))/delT ))
    print (nr, nf)
    reg1 = fn <= nr
    reg2 = (fn > nr) & (fn <= nf)
    reg3 = fn > nf

    # phs will be shaped (Q3, N2, N1, Q1)
    # so it will be soln_plane (Q3, None, None, Q1) *
    # some function shaped (None, N2, N1, None)
    phs = N.empty((Q3,N2,N1,Q1), N.float64)
    fr1 = N.power(( (T0/Tr)**2 + 2*fn*delT/Tr ), 0.5) * -n2sign[:,None]
    fr2 = N.ones((N2,N1)) * -n2sign[:,None]
    sqrt_term = 2 + 2*Tf/Tr - (T0/Tr)**2 - 2*fn*delT/Tr
    sqrt_term = N.where(sqrt_term < 0, 0, sqrt_term)
    fr3 = N.power(sqrt_term, 0.5) * -n2sign[:,None]
    box = N.empty((N2, N1), N.float64)
    box[reg1] = fr1[reg1]
    box[reg2] = fr2[reg2]
    box[reg3] = fr3[reg3]
    phs = soln_plane[:,None,None,:] * box[None,:,:,None]
    return phs

def kern_full(phs):
    Q3,N2,N1,Q1 = phs.shape
    n1_ax = N.linspace(-N1/2., N1/2., N1, endpoint=False)
    #n1_ax = N.arange(N1, dtype=N.float64)
    # fb is (N1, Q1)
    fb = -2.j*N.pi*n1_ax[:,None]*n1_ax[None,:]/N1
    # fz is (Q3, N2, N1, Q1) --> (Q3, N2, N1, NP1) upon IFFT
    zarg = 1.j*phs + fb[None,None,:,:]
    #fz = ifft(N.exp(zarg)) * N1
    fz = ifft(N.exp(zarg))
    return fz
