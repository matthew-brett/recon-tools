from Numeric import empty, sort
from pylab import angle, conjugate, Float, arange, take, zeros, mean, floor, \
     pi, sqrt, ones, sum, find, Int, resize, matrixmultiply, svd, transpose, \
     diag, putmask
from imaging.operations import Operation, Parameter
from imaging.util import ifft, apply_phase_correction, mod, linReg, checkerline
from imaging.punwrap import unwrap2D
    
class UnbalPhaseCorrection (Operation):

    def unwrap_volume(self, phases):
        """
        unwrap phases one "slice" at a time, where the volume
        is sliced along a single pe line (dimensions = nslice X n_fe)
        take care to move all the surfaces to roughly the same height
        @param phases is a volume of wrapped phases
        @return: uphases an unwrapped volume, shrunk to masked region
        """
        mask = zeros((self.refShape[0],self.refShape[2]))
        mask[:,self.lin1:self.lin2] = 1
        vol_shape = phases.shape
        uphases = empty(vol_shape, Float)
        from pylab import show, plot, imshow, title, colorbar
        for r in range(vol_shape[1]):
            uphases[:,r,:] = unwrap2D(phases[:,r,:], mask=mask)

        midsl, midln, midpt = (vol_shape[0]/2, vol_shape[1]/2, vol_shape[2]/2)
        # in what 2pi bin is the middle point of the volume?
        # if it is not close to zero, we'll want to shift it there,
        # and then shift the other surfaces appropriately
        height_at_midpt = floor(uphases[midsl,midln,midpt]/pi)
        # the point at uphases[0,midln,0] is from the same surface
        # as the middle point in the volume; this will be the reference
        correct_height = uphases[0,midln,self.refShape[2]-1] - 2*pi*height_at_midpt
        # take a sample from the corner of every unwrapped surface (across mu)
        # Due to the mask, these values will have unwrapped to be 0 + k2pi,
        # and can be used to move every surface back to the same offset
        # as the middle slice
        offsets = uphases[0,:,self.refShape[2]-1]
        offset_corrections = offsets - correct_height
        for r in range(vol_shape[1]):
            uphases[:,r,:] = uphases[:,r,:] - offset_corrections[r]

        from pylab import imshow, plot, show, title
        for z in range(vol_shape[0]):
            for r in range(0,vol_shape[1],2):
                plot(uphases[z,r])
            title("slice %d, even"%(z,))
            show()
            for r in range(1,vol_shape[1],2):
                plot(uphases[z,r])
            title("slice %d, odd"%(z,))
            show()
##             imshow(take(uphases[z], arange(2,vol_shape[1],2), axis=0))
##             title("slice %d even"%(z,))
##             colorbar()
##             show()
##             imshow(take(uphases[z], arange(1,vol_shape[1]-1,2), axis=0))
##             title("slice %d odd"%(z,))
##             colorbar()
##             show()
        return uphases[:,:,self.lin1:self.lin2]

    def masked_avg(self, S):
        """
        Take all even or odd lines in a slice, return the mean of these lines
        along with a mask, where the mask is set to zero if the standard
        deviation exceeds a threshold (taken to mean noisy data), and also
        a sum of residuals from the linear fit of each line (taken as a measure
        of linearity)
        @param S is all even or odd lines of a slice
        @return: E is the mean of these lines, mask is variance based mask, and
        sum(res) is the measure of linearity
        """
        
        nr,np = S.shape
        mask = ones((np,))

        b = empty((nr,), Float)
        m = empty((nr,), Float)
        res = empty((nr,), Float)
        for r in range(nr):
            #(_, _, res[r]) = linReg(arange(len(S[0])), S[r])
            (b[r], m[r], res[r]) = linReg(arange(len(S[0])), S[r])
        E = mean(S)
        std = sqrt(sum((S-E)**2)/nr)

##         from pylab import show, plot, title
##         color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
##         for r in range(nr):
##             plot(S[r], color[r%7])
##             plot(arange(len(S[0]))*m[r]+b[r], color[r%7]+'--')
##         plot(std, 'bo')
##     #    plot(E, 'go')
##         #title("slice = %d"%(snum,))
##         #print res
##         show()

        putmask(mask, std>1, 0)
        return E, mask, sum(res)

    def solve_phase(self, ev, odd, q_mask, z_mask):
        """let V = (beta_pr, beta, alpha_pr, alpha, e_pr, e)^T,
        we want to solve:
        phi(q,s,0) = q*(beta_pr + 2*beta) + s*(alpha_pr + 2*alpha) + e_pr + 2*e
        phi(q,s,1) = q*(beta_pr - 2*beta) + s*(alpha_pr - 2*alpha) + e_pr - 2*e
        with our overdetermined data.
        so P = [ev[0] odd[0] ev[1] odd[1] ... ev[S] odd[S]]^T
        for all selected slices
        A = [ ... ]
        and with AV = P, solve V = inv(A)P
        """
        BPR, B, ALPR, AL, EPR, E = (0,1,2,3,4,5)
        n_chunks = sum(z_mask)
        rows_in_chunk = sum(take(q_mask, find(z_mask)), axis=1)
        z_ind = find(z_mask)
        A = empty((sum(rows_in_chunk)*2, 6), Float)
        V = empty((sum(rows_in_chunk)*2, 1), Float)
        row_start, row_end = 0, 0
        for c in range(n_chunks):
            # alternate for even and odd rows
            for s in [0, 1]:
                row_start = row_end
                row_end = row_start + rows_in_chunk[c]
                q_ind = find(q_mask[z_ind[c]])
                V[row_start:row_end,0] = s and take(odd[z_ind[c]], q_ind) or take(ev[z_ind[c]], q_ind)
                A[row_start:row_end,BPR] = q_ind + self.lin1 #need to put back the truncated pixels
                A[row_start:row_end,B] = s and 2*(q_ind+self.lin1) or -2*(q_ind+self.lin1)
                A[row_start:row_end,ALPR] = z_ind[c]
                A[row_start:row_end,AL] = s and 2*z_ind[c] or - 2*z_ind[c]
                A[row_start:row_end,EPR] = 1
                A[row_start:row_end,E] = s and 2 or -2

        [u,s,vt] = svd(A)
        P = matrixmultiply(transpose(vt), matrixmultiply(diag(1/s), \
                                          matrixmultiply(transpose(u), V)))

        return tuple(P) 

    def correction_volume(self):
        """
        build the volume of phase correction lines with
        theta(s,r,q) = r*(Bpr*q + Apr*s + Epr) +/- (B*q + A*s + E)
        """
        (S, R, Q) = self.volShape
        (bpr, b, apr, a, epr, e) = self.coefs
        A = empty((R, len(self.coefs)), Float)
        B = empty((len(self.coefs), Q), Float)
        theta = empty(self.volShape, Float)
        # build B matrix, always stays the same
        B[0] = arange(Q)
        B[1] = arange(Q)
        B[2] = B[3] = B[4] = B[5] = 1
        # build A matrix, changes slightly as s varies
        zigzag = checkerline(R)
        r_line = arange(R) - R/2
        A[:,0] = bpr*r_line
        A[:,1] = b*zigzag
        A[:,4] = epr*r_line
        A[:,5] = e*zigzag
        for s in range(S):
            A[:,2] = s*apr*r_line
            A[:,3] = s*a*zigzag
            theta[s] = matrixmultiply(A,B)
        return theta

    def run(self, image):
        if not image.ref_data:
            self.log("No reference volume, quitting")
            return
        if len(image.ref_vols) > 1:
            self.log("Could be performing Balanced Phase Correction!")

        self.volShape = image.data.shape[1:]
        #1st copy in memory
        refVol = image.ref_data[0]
        n_slice, n_pe, n_fe = self.refShape = refVol.shape
        lin_radius = 50.0
        lin_pix = int(round(lin_radius/image.xsize))
        (self.lin1, self.lin2) = (lin_pix > n_fe/2) and (0,n_fe) or \
                                 ((n_fe/2-lin_pix), (n_fe/2+lin_pix))
        self.lin_fe = lin_pix*2
        take_order = arange(n_pe)+1
        take_order[-1] = 0
       
        inv_ref = ifft(refVol)
        inv_ref = conjugate(take(inv_ref, take_order, axis=1)) * inv_ref
       
        # comes back truncated to linear region:
        # from this point on, into the svd, work with truncated arrays
        # (still have "true" referrences from from self.refShape, self.lin1, etc)
        #phs_vol = self.unwrap_volume(angle(inv_ref[:,:,self.lin1:self.lin2]))
        phs_vol = self.unwrap_volume(angle(inv_ref))
        #set up some arrays
        phs_even = empty((n_slice, self.lin_fe), Float)
        phs_odd = empty((n_slice, self.lin_fe), Float)
        z_mask = zeros(n_slice)
        q_mask = zeros((n_slice, self.lin_fe))
        res = zeros((n_slice,), Float)
        for z in range(n_slice):
            phs_even[z], mask_e, res_e = \
                         self.masked_avg(take(phs_vol[z], arange(2,n_pe,2)))
            phs_odd[z], mask_o, res_o = \
                        self.masked_avg(take(phs_vol[z], arange(1,n_pe-1,2)))
            res[z] = res_e + res_o

            q_mask[z] = mask_e*mask_o
            
        sres = sort(res)
        selected = [find(res==c)[0] for c in sres[:4]]
        for c in selected: z_mask[c] = 1
        self.coefs = (beta_pr, beta, alpha_pr, alpha, e_pr, e) = \
                  self.solve_phase(phs_even, phs_odd, q_mask, z_mask)

##         print "computed coefficients:"
##         print "\tBpr: %f, B: %f, Apr: %f, A: %f, e_pr: %f, e: %f"\
##               %(beta_pr, beta, alpha_pr, alpha, e_pr, e)

## ##Uncomment this section to see how the computed lines fit the real lines
##         from pylab import title, plot, show
##         for z in range(n_slice):
##             plot(find(q_mask[z]), take(phs_even[z], find(q_mask[z])))
##             plot(find(q_mask[z]), take(phs_odd[z], find(q_mask[z])))
##             plot((arange(self.lin_fe)+self.lin1)*(beta_pr - 2*beta) + z*(alpha_pr - 2*alpha) + e_pr - 2*e, 'b--')
##             plot((arange(self.lin_fe)+self.lin1)*(beta_pr + 2*beta) + z*(alpha_pr + 2*alpha) + e_pr + 2*e, 'g--')
##             tstr = "slice %d"%(z)
##             if z_mask[z]: tstr += " (selected)"
##             title(tstr)
##             show()


        theta_vol = self.correction_volume()

        for dvol in image.data:
            dvol[:] = apply_phase_correction(dvol, theta_vol)
