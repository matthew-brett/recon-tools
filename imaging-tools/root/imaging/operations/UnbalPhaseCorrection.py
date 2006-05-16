from Numeric import empty, sort
from pylab import angle, conjugate, Float, arange, take, zeros, mean, floor, \
     pi, sqrt, ones, sum, find, Int, resize, matrixmultiply, svd, transpose, \
     diag, putmask, sign, asarray
from imaging.operations import Operation, Parameter
from imaging.util import ifft, apply_phase_correction, mod, linReg, shift
from imaging.punwrap import unwrap2D

class UnbalPhaseCorrection (Operation):
    """Unbalanced Phase Correction tries to determine six parameters which
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
                  description="Radius of the region of greatest linearity "\
                  "within the magnetic field, in mm (normally 70-80mm)"),
        )

    def unwrap_volume(self, phases):
        """
        unwrap phases one "slice" at a time, where the volume
        is sliced along a single pe line (dimensions = nslice X n_fe)
        take care to move all the surfaces to roughly the same height
        @param phases is a volume of wrapped phases
        @return: uphases an unwrapped volume, shrunk to masked region
        """
        zdim,ydim,xdim = vol_shape = phases.shape
        #mask = ones((zdim,xdim))
        #mask = zeros((zdim,xdim))
        #mask[:,self.lin1:self.lin2] = 1
        uphases = empty(vol_shape, Float)
        midsl, midpt = (vol_shape[0]/2, vol_shape[2]/2)
        #from pylab import show, plot, imshow, title, colorbar, figure
        # unwrap the volume sliced along each PE line
        # the middle of each surface should be between -pi and pi,
        # if not, put it there!
        for r in range(0,vol_shape[1],1):
            uphases[:,r,:] = unwrap2D(phases[:,r,:])
            height = uphases[midsl,r,midpt]
            height = int((height+sign(height)*pi)/2/pi)
            uphases[:,r,:] = uphases[:,r,:] - 2*pi*height
                    
        return uphases[:,:,self.lin1:self.lin2]

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
        
        nr,np = S.shape
        mask = ones((np,))

        #b = empty((nr,), Float)
        #m = empty((nr,), Float)
        res = empty((nr,), Float)
        for r in range(nr):
            (_, _, res[r]) = linReg(arange(len(S[0])), S[r])
            #(b[r], m[r], res[r]) = linReg(arange(len(S[0])), S[r])
        E = mean(S)
        std = sqrt(sum((S-E)**2)/nr)

##         from pylab import show, plot, title
##         color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
##         for r in range(nr):
##             plot(S[r], color[r%7])
##             #plot(arange(len(S[0]))*m[r]+b[r], color[r%7]+'--')
##         plot(std, 'bo')
##         plot(E, 'go')
##         #title("slice = %d"%(snum,))
##         #print res
##         show()

        putmask(mask, std>1, 0)
        return E, mask, sum(res)

    def solve_phase(self, pos, neg, q_mask, z_mask):
        """let V = (a1 a2 a3 a4 a5 a6)^T,
        we want to solve:
        phi(s,u,r) =  2[rA1 + sA3 + A5] - [rA2 + sA4 + A6] u-pos
        phi(s,u,r) = -2[rA1 + sA3 + A5] - [rA2 + sA4 + A6] u-neg
        with our overdetermined data.
        so P = [pos[0] neg[0] pos[1] neg[1] ... pos[S] neg[S]]^T
        for all selected slices, and similarly
        A = [2r0 -r0 2s0 -s0 2 -1;
             -2r1 -r1 -2s0 -s0 -2 -1;
             ...
             2r0 -r0 2s1 -s1 2 -1;
             ...
             2r0 -r0 2s2 -s2 2 -1;
             -2r1 -r1 -2s2 -s2 -2 -1;
             2r2 -r2 2s2 -s2 2 -1;
             ...]
        Then with AV = P, solve V = inv(A)P
        """

        A1, A2, A3, A4, A5, A6 = (0,1,2,3,4,5)
        n_chunks = sum(z_mask)
        rows_in_chunk = sum(take(q_mask, find(z_mask)), axis=1)
        z_ind = find(z_mask)
        A = empty((sum(rows_in_chunk)*2, 6), Float)
        P = empty((sum(rows_in_chunk)*2, 1), Float)
        row_start, row_end = 0, 0
        for c in range(n_chunks):
            # alternate for pos and neg rows
            for b in [-1, 1]:
                row_start = row_end
                row_end = row_start + rows_in_chunk[c]
                q_ind = find(q_mask[z_ind[c]])
                P[row_start:row_end,0] = b==-1 and take(neg[z_ind[c]], q_ind) \
                                               or  take(pos[z_ind[c]], q_ind)
                # note these q_ind values are made relative to real fe points:
                A[row_start:row_end,A1] = b*2*(q_ind + self.lin1)
                A[row_start:row_end,A2] = -(q_ind + self.lin1)
                A[row_start:row_end,A3] = b*2*z_ind[c]
                A[row_start:row_end,A4] = -z_ind[c]
                A[row_start:row_end,A5] = b*2
                A[row_start:row_end,A6] = -1
                
        [u,s,vt] = svd(A)
        V = matrixmultiply(transpose(vt), matrixmultiply(diag(1/s), \
                                          matrixmultiply(transpose(u), P)))

        return tuple(V) 

    def correction_volume(self, u_pos, u_neg):
        """
        build the volume of phase correction lines with
        theta(s,u,r) = u*[r*A2 + s*A4 + A6] + (-1)^u*[r*A1 + s*A3 + A5]
        or some variation (see doc)
        A is (n_pe x 6) = 
        B is (6 x n_fe) = [0:N; 0:N; 1; 1; 1; 1]
        """
        (S, U, R) = self.volShape
        (a1, a2, a3, a4, a5, a6) = self.coefs
        A = empty((U, len(self.coefs)), Float)
        B = empty((len(self.coefs), R), Float)
        theta = empty(self.volShape, Float)
        # build B matrix, always stays the same
        B[0] = arange(R)
        B[1] = arange(R)
        B[2] = B[3] = B[4] = B[5] = 1
        # u_line & zigzag define how the correction changes per PE line
        zigzag = empty(U)
        for r in range(U/2):
            zigzag[u_pos[r]] = 1
            zigzag[u_neg[r]] = -1
        u_line = empty(U)
        for r in range(self.xleave):
            u_line[r:U:self.xleave] = arange(U/self.xleave)-U/(2*self.xleave)
        # if data is multishot centric, must count mu=0 twice, ie u_line is:
        # [-U/2 + 1 ... 0] U [0 ... U/2 - 1], zigzag = [-1,1,...,1,1,...,1,-1]
        if self.iscentric:
            u_line[0:U/2] = -1*(u_line[0:U/2] + 1)
            zigzag[0:U/2] *= -1
        # build A matrix, changes slightly as s varies
        A[:,0] = a2*u_line
        A[:,1] = a1*zigzag
        A[:,4] = a6*u_line
        A[:,5] = a5*zigzag
        for s in range(S):
            # these are the slice-dependent columns
            A[:,2] = s*a4*u_line
            A[:,3] = s*a3*zigzag
            theta[s] = matrixmultiply(A,B)
        return theta

    def run(self, image):
        if not image.ref_data:
            self.log("No reference volume, quitting")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        self.volShape = image.data.shape[1:]
        # set up data indexing helpers, based on acquisition order.
        # iscentric says whether kspace is multishot centric;
        # xleave is the factor to which kspace data has been interleaved
        # (in the case of multishot interleave)
        self.iscentric = image.petable_name.find('cen') > 0 or \
                         image.petable_name.find('alt') > 0
        self.xleave = self.iscentric and 1 or image.nseg
        
        refVol = image.ref_data[0]
        n_slice, n_pe, n_fe = self.refShape = refVol.shape

        # [self.lin1:self.lin2] is the linear region of the gradient
        # self.lin_fe is the # of points in this region
        lin_pix = int(round(self.lin_radius/image.xsize))
        (self.lin1, self.lin2) = (lin_pix > n_fe/2) and (0,n_fe) or \
                                 ((n_fe/2-lin_pix), (n_fe/2+lin_pix))
        self.lin_fe = self.lin2-self.lin1

##         #reverse centric neg rows
##         trev = -1 - arange(n_pe)
##         #image.data[:,:,:n_pe/2,:]=take(image.data[:,:,:n_pe/2:], trev, axis=-1)
##         refVol[:,:n_pe/2,:] = take(refVol[:,:n_pe/2,:], trev, axis=-1)
        
        conj_order = arange(n_pe)
        shift(conj_order, 0, -self.xleave)
        inv_ref = ifft(refVol)
        inv_ref = conjugate(take(inv_ref, conj_order, axis=1)) * inv_ref

        # pos_order, neg_order define which rows in a slice are grouped
        pos_order = zeros(n_pe/2)
        neg_order = zeros(n_pe/2)
        for n in range(self.xleave):
            pos_order[n:n_pe/2:self.xleave] = arange(n,n_pe,2*self.xleave)
            neg_order[n:n_pe/2:self.xleave] = arange(self.xleave+n,n_pe,2*self.xleave)
       
        # comes back truncated to linear region:
        # from this point on, into the svd, work with truncated arrays
        # (still have "true" referrences from from self.refShape, self.lin1, etc)
        phs_vol = self.unwrap_volume(angle(inv_ref))
        #set up some arrays
        if self.iscentric:
            # centric has 4 distinct profile (2 sets of symmetries)
            phs_pos_upper = empty((n_slice, self.lin_fe), Float)
            phs_pos_lower = empty((n_slice, self.lin_fe), Float)
            phs_neg_upper = empty((n_slice, self.lin_fe), Float)
            phs_neg_lower = empty((n_slice, self.lin_fe), Float)
        else:
            phs_pos = empty((n_slice, self.lin_fe), Float)
            phs_neg = empty((n_slice, self.lin_fe), Float)
        
        z_mask = zeros(n_slice)
        q_mask = zeros((n_slice, self.lin_fe))
        res = zeros((n_slice,), Float)
        if self.iscentric:
            for z in range(n_slice):
                # seems that for upper, skip 1st pos and last neg; 
                #            for lower, skip last pos and last neg
                phs_pos_upper[z], mask_pu, res_pu = \
                       self.masked_avg(take(phs_vol[z], \
                                            pos_order[n_pe/4+1:]))
                
                phs_neg_upper[z], mask_nu, res_nu = \
                       self.masked_avg(take(phs_vol[z], \
                                            neg_order[n_pe/4:n_pe/2-1]))
                
                phs_pos_lower[z], mask_pl, res_pl = \
                       self.masked_avg(take(phs_vol[z], neg_order[:n_pe/4-1]))
                
                phs_neg_lower[z], mask_nl, res_nl = \
                       self.masked_avg(take(phs_vol[z], pos_order[:n_pe/4-1]))
                
                res[z] = res_pu + res_nu + res_pl + res_nl
                q_mask[z] = mask_pu*mask_nu*mask_pl*mask_nl
        else:
            for z in range(n_slice):
                # for pos_order, skip 1st 2 for xleave=2
                # for neg_order, skip last 2 for xleave=2
                phs_pos[z], mask_p, res_p = \
                            self.masked_avg(take(phs_vol[z], \
                                            pos_order[self.xleave:]))
                phs_neg[z], mask_n, res_n = \
                            self.masked_avg(take(phs_vol[z], \
                                            neg_order[:-self.xleave]))

                res[z] = res_p + res_n
                q_mask[z] = mask_p*mask_n

        # find 4 slices with smallest residual
        sres = sort(res)
        selected = [find(res==c)[0] for c in sres[:6]]
        for c in selected:
            z_mask[c] = 1
            if(sum(q_mask[c]) == 0):
                self.log("Could not find enough slices with sufficiently uniform\n"\
                "phase profiles. Try shortening the lin_radius parameter to\n"\
                "unwrap a less noisy region of the image phase.\n"\
                "Current FOV: %fmm, Current lin_radius: %fmm\n"%(n_fe*image.xsize,self.lin_radius))
                return
        
        if self.iscentric:
            self.coefs = self.solve_phase(phs_pos_upper, phs_neg_upper, q_mask, z_mask)
            print self.coefs
            theta_upper = self.correction_volume(pos_order, neg_order)
            # can use same inverse matrix to solve for lower half,
            # but transform with diag([1, -1, 1, -1, 1, -1])
            v = self.solve_phase(phs_pos_lower, phs_neg_lower, q_mask, z_mask)
            self.coefs = tuple(matrixmultiply(diag([1, -1, 1, -1, 1, -1]), \
                                              asarray(v)))
            print self.coefs        
            theta_lower = self.correction_volume(pos_order, neg_order)

            theta_vol = empty(theta_lower.shape, theta_lower.typecode())
            theta_vol[:,:n_pe/2,:] = theta_lower[:,:n_pe/2,:]
            theta_vol[:,n_pe/2:,:] = theta_upper[:,n_pe/2:,:]
            
        else:
            self.coefs = (a1,a2,a3,a4,a5,a6) = \
                         self.solve_phase(phs_pos, phs_neg, q_mask, z_mask)
            print self.coefs
            theta_vol = self.correction_volume(pos_order, neg_order)            

        for dvol in image.data:
            dvol[:] = apply_phase_correction(dvol, -theta_vol)

##         print "computed coefficients:"
##         print "\ta1: %f, a2: %f, a3: %f, a4: %f, a5: %f, a6: %f"\
##               %self.coefs

## ## Uncomment this section to see how the computed lines fit the real lines
##         from pylab import title, plot, show
##         for z in range(n_slice):
##             plot(find(q_mask[z]), take(phs_pos[z], find(q_mask[z])))
##             plot(find(q_mask[z]), take(phs_neg[z], find(q_mask[z])))
##             plot((arange(self.lin_fe)+self.lin1)*(-a2 - 2*a1) + z*(-a4 - 2*a3) - a6 - 2*a5, 'g--')
##             plot((arange(self.lin_fe)+self.lin1)*(-a2 + 2*a1) + z*(-a4 + 2*a3) - a6 + 2*a5, 'b--')
##             tstr = "slice %d"%(z)
##             if z_mask[z]: tstr += " (selected)"
##             title(tstr)
##             show()

            

