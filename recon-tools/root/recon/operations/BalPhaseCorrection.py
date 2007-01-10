"Applies a Balanced Phase Correction on data sets with two reference volumes"

import Numeric as N
from MLab import angle, diag
from LinearAlgebra import singular_value_decomposition as svd
import pylab as P

from recon.operations import Operation, Parameter, verify_scanner_image
from recon.util import ifft, apply_phase_correction, linReg, checkerline, \
     unwrap_ref_volume, reverse

def shift_columns_left(matrix):

    for sl in matrix:
        for col in range(sl.shape[1]-1):
            sl[:,col] = sl[:,col+1]
        sl[:,sl.shape[1]-1] = N.zeros(sl.shape[0]).astype(sl.typecode())
        
def shift_columns_right(matrix):
    """shifts columns (assumed to be last dimension) right w/o wrap-around"""
    for sl in matrix:
        for col in reverse(N.arange(sl.shape[1]-1)):
            sl[:,col] = sl[:,col-1]
        sl[:,0] = N.zeros(sl.shape[0]).astype(sl.typecode())

class BalPhaseCorrection (Operation):
    params = (
        Parameter(name="lin_radius", type="float", default=70.0,
                  description="Radius of the region of greatest linearity "\
                  "within the magnetic field, in mm (normally 70-80mm)"),
        )

    def run(self, image):
        if not verify_scanner_image(self, image): return
        if not hasattr(image, "ref_data") or len(image.ref_vols) < 2:
            self.log("Not enough reference volumes, quitting.")
            return

        self.volShape = image.shape[-3:]
        
        #shift_columns_left(image.ref_data[0])
        #shift_columns_right(image.ref_data[0])
        inv_ref0 = ifft(image.ref_data[0])
        shift_columns_left(image.ref_data[1])
        #shift_columns_right(image.ref_data[1])
        inv_ref1 = ifft(image.ref_data[1])
        inv_ref = inv_ref0*N.conjugate(inv_ref1)
        
        n_slice, n_pe, n_fe = self.refShape = inv_ref0.shape

        lin_pix = int(round(self.lin_radius/image.xsize))
        (self.lin1, self.lin2) = (lin_pix > n_fe/2) and (0,n_fe) or \
                                 ((n_fe/2-lin_pix), (n_fe/2+lin_pix))
        self.lin_fe = self.lin2-self.lin1
        
        # comes back smaller! read direction goes from lin1:lin2
        phs_vol = unwrap_ref_volume(angle(inv_ref), self.lin1, self.lin2)
        #phs_vol = angle(inv_ref)[...,self.lin1:self.lin2]
        
        s_mask = N.zeros(n_slice) # this will be 4 most "linear" slices
        r_mask = N.ones((self.lin_fe)) #
        u_mask = N.ones((n_pe))  # 
        u_mask[0] = 0
        #u_mask[46:] = 0
        res = N.zeros((n_slice,), N.Float)
        for s in range(n_slice):
            for rn,row in enumerate(phs_vol[s]):
                (_, _, r) = linReg(row, X=N.arange(self.lin_fe))
                res[s] += r

        # find 4 slices with smallest residual
        sres = N.sort(res)
        selected = [N.nonzero(res==c)[0] for c in sres[:4]]
        for c in selected:
            s_mask[c] = 1
        
        self.coefs = (b1,b7,b8,b5,b6) = \
                         self.solve_phase(phs_vol, r_mask, u_mask, s_mask)
        #print self.coefs
        Tl = image.T_pe
        delT = image.delT
        #print "Tl = %f; delT = %f"%(Tl,delT,)
        theta_vol = self.correction_volume(Tl, delT)

        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = apply_phase_correction(image[:], -theta_vol)
        else:
            for dvol in image:
                dvol[:] = apply_phase_correction(dvol[:], -theta_vol)



    def solve_phase(self, pvol, r_mask, u_mask, s_mask):
        """let V = (b1 b7 b8 b5 b6)^T,
        we want to solve:
        0.5*phi(s,u,r) = (b1 + rb7 + sb8)(-1)^u + (rb5 + sb6)u
        with our overdetermined data.
        so P = [pvol[s=0,u=0,:] pvol[s=0,u=1,:] ... pvol[s=S,u=U,:]^T
        for all S selected slices, and similarly
        A = [1 r0 s0 u0*r0 u0*s0;
             1 r1 s0 u0*r1 u0*s0;
             ...
             1 rR s0 u0*rR u0*s0;
             ---------------------
            -1 -r0 -s0 u1*r0 u1*s0;
            ...
            -1 -rR -s0 u1*rR u1*s0;
             ----------------------
             ...
             1 rR sS uU*rR uU*sS]
        
        Then with AV = P, solve V = inv(A)P
        """
        (B1,B7,B8,B5,B6) = (0,1,2,3,4)
        U,R = self.refShape[-2:]
        r_ind = N.nonzero(r_mask) # this is w/ resp. to truncated row size
        u_ind = N.nonzero(u_mask)
        s_ind = N.nonzero(s_mask)
        n_chunks = len(s_ind) # 1 chunk per slice(?)
        n_parts = len(u_ind)  # 1 part per pe-line(?)
        A = N.empty((n_chunks*n_parts*len(r_ind), 5), N.Float)
        P = N.empty((n_chunks*n_parts*len(r_ind), 1), N.Float)
        row_start, row_end = 0, 0
        #for c in range(n_chunks):
        #    for p in range(n_parts):
        for s in s_ind:
            for u in u_ind:
                sign = 1 - 2*(u%2)
                row_start = row_end
                row_end = row_start + len(r_ind)
                # the data vector
                P[row_start:row_end,0] = 0.5*N.take(pvol[s, u, :], r_ind)
                
                # the phase-space vector
                A[row_start:row_end,B1] = sign
                A[row_start:row_end,B7] = sign*(r_ind+self.lin1-R/2) # un-truncated
                A[row_start:row_end,B8] = sign*s
                A[row_start:row_end,B5] = (u-U/2)*(r_ind+self.lin1-R/2) # ditto
                A[row_start:row_end,B6] = (u-U/2)*s
                
##         f = open("matrix", "w")
##         f.write("[b1 r*b7 s*b8 u*r*b5 u*s*b6]\n")
##         for row in A:
##             f.write("[%d %d %d %d %d]\n"%(tuple(row)))
##         f.close()
        
        [u,s,vt] = svd(A)
        V = N.dot(N.transpose(vt), N.dot(diag(1/s), N.dot(N.transpose(u), P)))

##         foo = reshape(matrixmultiply(A,V), (n_chunks, n_parts, len(r_ind)))
##         from pylab import figure
##         print s_ind
##         print u_ind
##         print r_ind
##         for s in range(n_chunks):
##             for u in range(n_parts):
##                 plot(foo[s,u])
##             figure()
##             for u in range(n_parts):
##                 plot(take(pvol[s_ind[s],u_ind[u]], r_ind)/2.)
##             show()
    
        
        return tuple(V) 

    def correction_volume(self, Tl, delT):
        """
        build the volume of phase correction lines with
        theta(s,u,r) = u*[r*B5 + s*B6] + (-1)^u*[B1 + r*B7 + s*B8]

        A is (n_pe x 8) = 
        B is (8 x n_fe) = [0:N; 0:N; 1; 1; 1; 1]
        """
        (S, M, R) = self.volShape
        (b1,b7,b8,b5,b6) = self.coefs
        print delT, Tl
        b2 = (delT/Tl)*b5[0]
        b3 = (delT/Tl)*b6[0]
        b4 = (Tl/delT)*b1[0]
        print `["%2.8f"%flo for flo in [b1,b2,b3,b4,b5,b6,b7,b8]]`
        b2=b3=b4 = 0.0
        A = N.empty((M, 8), N.Float)
        B = N.empty((8, R), N.Float)
        theta = N.empty(self.volShape, N.Float)

        # build B matrix, always stays the same
        B[0,:] = b1[0]
        B[1,:] = (N.arange(R)-R/2)*b2
        B[2,:] = (N.arange(R)-R/2)*b7[0]
        B[3,:] = b3
        B[4,:] = b8[0]
        B[5,:] = b4
        B[6,:] = (N.arange(R)-R/2)*b5[0]
        B[7,:] = b6[0]
        
        # u_line & zigzag define how the correction changes per PE line
        m_line = N.arange(M)
        zigzag = checkerline(M)
        # build A matrix, changes slightly as s varies
        A[:,0] = zigzag
        A[:,1] = zigzag
        A[:,2] = zigzag
        A[:,5] = m_line
        A[:,6] = m_line
        for s in range(S):
            A[:,3] = s*zigzag
            A[:,4] = s*zigzag
            A[:,7] = s*m_line
            theta[s] = N.dot(A,B)
        return theta


