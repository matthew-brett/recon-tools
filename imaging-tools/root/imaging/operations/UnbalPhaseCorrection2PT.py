from Numeric import empty, sort
from pylab import angle, conjugate, Float, arange, take, zeros, mean, \
     pi, sqrt, ones, sum, find, Int, matrixmultiply, svd, transpose, \
     diag, putmask, asarray
from imaging.operations import Operation, Parameter
from imaging.util import ifft, apply_phase_correction, mod, linReg, shift, \
     unwrap_ref_volume

class UnbalPhaseCorrection2PT (Operation):
    """Unbalanced Phase Correction tries to determine six parameters which
    define the shape of reference scan phase lines via an SVD solution. Using
    these parameters, a volume of correction phases is built and applied. The
    solution of the parameters changes slightly with different acquisition
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
#-----------------------------------------------------------------------------
    def masked_avg(self, S):
        """Take all pos or neg lines in a slice, return the mean of these lines
        along with a mask, where the mask is set to zero if the standard
        deviation exceeds a threshold (taken to mean noisy data), and also
        a sum of residuals from the linear fit of each line (taken as a measure
        of linearity)
        @param S: is all pos or neg lines of a slice
        @return: E is the mean of these lines, mask is variance based mask, and
        sum(res) is the measure of linearity
        """
        nr,np = S.shape
        mask = ones((np,))
        res = empty((nr,), Float)
        
        E = mean(S)
        std = sqrt(sum((S-E)**2)/nr)
        putmask(mask, std>1, 0)

##         # UNCOMMENT FOR PLOTTING
##         from pylab import show, plot, title
##         color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
##         for r in range(nr):
##             plot(S[r], color[r%7])
##         plot(std, 'bo')
##         plot(E, 'go')
##         show()
        
        x_ax = find(mask)
        if sum(mask)>2:
            for r in range(nr):
                (_, _, res[r]) = linReg(x_ax, take(S[r], x_ax))
        else: res = 1e10*ones(nr)

        return E, mask, sum(res)

#-----------------------------------------------------------------------------

    def solve_phase2(self, pos, neg, s_mask):
        #A1, A2, B1, B2 = (0,1,2,3)
        A1,B1 = (0,1)
        n_rows = sum(s_mask)
        s_ind = find(s_mask)
        #A = empty((n_rows*2, 4), Float)
        A = empty((n_rows*2, 2), Float)
        P = empty((n_rows*2, 1), Float)
        row_start, row_end = 0,0
        for b in [-1, 1]:
            row_start = row_end
            row_end = row_start + n_rows
            P[row_start:row_end,0] = b==-1 and take(neg, s_ind) \
                                           or  take(pos, s_ind)
            A[row_start:row_end,A1] = b*2
            #A[row_start:row_end,A2] = b*2*s_ind
            A[row_start:row_end,B1] = -1
            #A[row_start:row_end,B2] = -s_ind
        [u,s,vt] = svd(A)
        V = matrixmultiply(transpose(vt), matrixmultiply(diag(1/s), \
                                          matrixmultiply(transpose(u), P)))

        return tuple(V)
    
#-----------------------------------------------------------------------------
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
        # self.lin_fe is the number of points in this region
        lin_pix = int(round(self.lin_radius/image.xsize))
        (self.lin1, self.lin2) = (lin_pix > n_fe/2) and (0,n_fe) or \
                                 ((n_fe/2-lin_pix), (n_fe/2+lin_pix))
        self.lin_fe = self.lin2-self.lin1

        conj_order = arange(n_pe)
        if self.iscentric:
            shift(conj_order[:n_pe/2],0,1)
            shift(conj_order[n_pe/2:],0,-1)
        else:
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
        # (still have "true" referrences from self.refShape, self.lin1, etc)
        phs_vol = unwrap_ref_volume(angle(inv_ref), self.lin1, self.lin2)

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
        
        s_mask = zeros(n_slice)
        r_mask = zeros((n_slice, self.lin_fe))
        res = zeros((n_slice,), Float)
        ### FIND THE MEANS FOR EACH SLICE
        if self.iscentric:
            for s in range(n_slice):
                # seems that for upper, skip 1st pos and last neg; 
                #            for lower, skip last pos and last neg
                # pos_order, neg_order naming scheme breaks down here!
                # because for mu < 0 rows, where ref volume is upside-down,
                # ODD rows go "positive"
                # try to fix this some time
                phs_pos_upper[s], mask_pu, res_pu = \
                       self.masked_avg(take(phs_vol[s], \
                                            pos_order[n_pe/4+1:]))
                
                phs_neg_upper[s], mask_nu, res_nu = \
                       self.masked_avg(take(phs_vol[s], \
                                            neg_order[n_pe/4:n_pe/2-1]))
                
                phs_pos_lower[s], mask_pl, res_pl = \
                       self.masked_avg(take(phs_vol[s], neg_order[:n_pe/4-1]))
                
                phs_neg_lower[s], mask_nl, res_nl = \
                       self.masked_avg(take(phs_vol[s], pos_order[1:n_pe/4]))
                
                res[s] = res_pu + res_nu + res_pl + res_nl
                r_mask[s] = mask_pu*mask_nu*mask_pl*mask_nl
        else:
            for s in range(n_slice):
                # for pos_order, skip 1st xleave rows
                # for neg_order, skip last xleave rows
                phs_pos[s], mask_p, res_p = \
                            self.masked_avg(take(phs_vol[s], \
                                            pos_order[self.xleave:]))
                phs_neg[s], mask_n, res_n = \
                            self.masked_avg(take(phs_vol[s], \
                                            neg_order[:-self.xleave]))

                res[s] = res_p + res_n
                r_mask[s] = mask_p*mask_n

        # find 4 slices with smallest residual
        sres = sort(res)
        print sres
        selected = [find(res==c)[0] for c in sres[:4]]
        print selected
##         from pylab import plot, show, subplot, title
##         for c in selected:
##             subplot(211)
##             for row in take(phs_vol[c], pos_order[self.xleave:]):
##                 plot(row)
##             plot(r_mask[c], 'bo')
##             subplot(212)
##             title("slice %d, residual = %f"%(c,res[c]))
##             for row in take(phs_vol[c], neg_order[:-self.xleave]):
##                 plot(row)
##             plot(r_mask[c], 'bo')
##             show()
            
        for c in selected:
            s_mask[c] = 1
            if(sum(r_mask[c]) == 0):
                self.log("Not enough slices with sufficiently uniform\n"\
                "phase profiles. Try shortening the lin_radius parameter to\n"\
                "unwrap a less noisy region of the image phase.\n"\
                "Current FOV: %fmm, Current lin_radius: %fmm\n"\
                         %(n_fe*image.xsize,self.lin_radius))
                return
        ### SOLVE FOR THE SYSTEM PARAMETERS FOR UNMASKED SLICES
        if self.iscentric:
            # want to correct both segments "separately" (by splicing 2 thetas)
            v = self.solve_phase(phs_pos_upper, phs_neg_upper, \
                                          r_mask, s_mask)
            self.coefs = tuple(matrixmultiply(diag([1,1,1,1,1,1]),asarray(v)))
            print self.coefs
            theta_upper = self.correction_volume(pos_order, neg_order)

            # THIS IS IN FLUX:
            # can use same inverse matrix to solve for lower half,
            # but need to transform it with diag([1, -1, 1, -1, 1, -1])
            v = self.solve_phase(phs_pos_lower, phs_neg_lower, r_mask, s_mask)
##             self.coefs = tuple(matrixmultiply(diag([1, -1, 1, -1, 1, -1]), \
##                                               asarray(v)))
            self.coefs = v
            print self.coefs        
            theta_lower = self.correction_volume(pos_order, neg_order)

            theta_vol = empty(theta_lower.shape, theta_lower.typecode())
            theta_vol[:,:n_pe/2,:] = theta_lower[:,:n_pe/2,:]
            theta_vol[:,n_pe/2:,:] = theta_upper[:,n_pe/2:,:]
            
        else:
##             self.coefs = self.solve_phase(phs_pos, phs_neg, r_mask, s_mask)
##             theta_vol = self.correction_volume(pos_order, neg_order)

            a1 = empty((n_fe,), Float)
            #a2 = empty((n_fe,), Float)
            b1 = empty((n_fe,), Float)
            #b2 = empty((n_fe,), Float)
            # recompute means
            phs_vol = unwrap_ref_volume(angle(inv_ref), 0, n_fe)
            phs_pos = empty((n_slice, n_fe), Float)
            phs_neg = empty((n_slice, n_fe), Float)
            for s in range(n_slice):
                
                phs_pos[s] = mean(take(phs_vol[s], \
                                            pos_order[self.xleave:]))
                phs_neg[s] = mean(take(phs_vol[s], \
                                            neg_order[:-self.xleave]))
            for r in range(n_fe):
##                 (a1[r], a2[r], b1[r], b2[r]) = \
##                         self.solve_phase2(phs_pos[:,r], phs_neg[:,r], s_mask)
                (a1[r], b1[r]) = \
                        self.solve_phase2(phs_pos[:,r], phs_neg[:,r], s_mask)

            from imaging.util import checkerline
            theta_vol = empty(self.volShape, Float)
            m_line = arange(n_pe)-n_pe/2
            s_line = arange(n_slice)
            zigzag = checkerline(n_pe)
            for s in range(n_slice):
                for m in range(n_pe):
                    for r in range(n_fe):
##                         theta_vol[s,m,r] = m_line[m]*(b1[r]+s*b2[r]) + \
##                                            zigzag[m]*(a1[r]+s*a2[r])
                        theta_vol[s,m,r] = m_line[m]*(b1[r]) + \
                                           zigzag[m]*(a1[r])

##             from pylab import show, subplot, title, plot
##             for s in range(n_slice):
##                 subplot(2,1,1)
##                 plot(phs_pos[s])
##                 subplot(2,1,2)
##                 plot(phs_neg[s])
##                 title("slice #%d"%(s,))
##                 show()
##                 subplot(2,1,1)
##                 for r in range(0,n_pe,2):
##                     plot(theta_vol[s,r])
##                 subplot(2,1,2)
##                 for r in range(1,n_pe,2):
##                     plot(theta_vol[s,r])
##                 title("correction slice #%d"%(s,))
##                 show()

        for dvol in image.data:
            dvol[:] = apply_phase_correction(dvol, -theta_vol)

##         print "computed coefficients:"
##         print "\ta1: %f, a2: %f, a3: %f, a4: %f, a5: %f, a6: %f"\
##               %self.coefs

## ## Uncomment this section to see how the computed lines fit the real lines
##         (a1,a2,a3,a4,a5,a6) = self.coefs
##         from pylab import title, plot, show
##         for z in range(n_slice):
##             plot(find(r_mask[z]), take(phs_pos[z], find(r_mask[z])))
##             plot(find(r_mask[z]), take(phs_neg[z], find(r_mask[z])))
##             plot((arange(self.lin_fe)+self.lin1)*(-a2 - 2*a1) + z*(-a4 - 2*a3) - a6 - 2*a5, 'g--')
##             plot((arange(self.lin_fe)+self.lin1)*(-a2 + 2*a1) + z*(-a4 + 2*a3) - a6 + 2*a5, 'b--')
##             tstr = "slice %d"%(z)
##             if s_mask[z]: tstr += " (selected)"
##             title(tstr)
##             show()

            

