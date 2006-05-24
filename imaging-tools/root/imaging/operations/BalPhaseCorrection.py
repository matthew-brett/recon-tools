"Applies a Balanced Phase Correction on data sets with two reference volumes"

#from FFT import inverse_fft
from Numeric import sort
from pylab import angle, conjugate, sin, cos, Complex32, Complex, fft, arange, reshape, ones, sqrt, plot, mean, take, pi, zeros, Float, show, floor, median, NewAxis, transpose, matrixmultiply, svd, exp, Int, diag, title, asarray, legend, empty, sign, putmask, find, sum
from imaging.operations import Operation, Parameter
from imaging.util import shift, fft, ifft, apply_phase_correction, unwrap1D, linReg, mod, checkerline, unwrap_ref_volume

def shift_columns_left(matrix):

    for slice in matrix:
        for col in range(slice.shape[1]-1):
            slice[:,col] = slice[:,col+1]
        slice[:,slice.shape[1]-1] = zeros(slice.shape[0]).astype(slice.typecode())
        
def shift_columns_right(matrix):
    """shifts columns (assumed to be last dimension) right w/o wrap-around"""
    for slice in matrix:
        for col in (-1 - arange(slice.shape[1]-1)):
            slice[:,col] = slice[:,col-1]
        slice[:,0] = zeros(slice.shape[0]).astype(slice.typecode())

class BalPhaseCorrection (Operation):
    params = (
        Parameter(name="lin_radius", type="float", default=70.0,
                  description="Radius of the region of greatest linearity "\
                  "within the magnetic field, in mm (normally 70-80mm)"),
        )

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
        r_ind = find(r_mask) # this is w/ resp. to truncated row size
        u_ind = find(u_mask)
        s_ind = find(s_mask)
        n_chunks = len(s_ind) # 1 chunk per slice(?)
        n_parts = len(u_ind)  # 1 part per pe-line(?)
        A = empty((n_chunks*n_parts*len(r_ind), 5), Float)
        P = empty((n_chunks*n_parts*len(r_ind), 1), Float)
        row_start, row_end = 0, 0
        #for c in range(n_chunks):
        #    for p in range(n_parts):
        for s in s_ind:
            for u in u_ind:
                sign = 1 - 2*(u%2)
                row_start = row_end
                row_end = row_start + len(r_ind)
                # the data vector
                P[row_start:row_end,0] = 0.5*take(pvol[s, u, :], r_ind)
                
                # the phase-space vector
                A[row_start:row_end,B1] = sign
                A[row_start:row_end,B7] = sign*(r_ind+self.lin1-32) # un-truncated
                A[row_start:row_end,B8] = sign*s
                A[row_start:row_end,B5] = (u)*(r_ind+self.lin1-32) # ditto
                A[row_start:row_end,B6] = (u)*s
                
        f = open("matrix", "w")
        f.write("[b1 r*b7 s*b8 u*r*b5 u*s*b6]\n")
        for row in A:
            f.write("[%d %d %d %d %d]\n"%(tuple(row)))
        f.close()
        
        [u,s,vt] = svd(A)
        V = matrixmultiply(transpose(vt), matrixmultiply(diag(1/s), \
                                          matrixmultiply(transpose(u), P)))

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
        (S, U, R) = self.volShape
        (b1,b7,b8,b5,b6) = self.coefs
        b2 = (delT/Tl)*b5[0]
        b3 = (delT/Tl)*b6[0]
        b4 = (Tl/delT)*b1[0]
        A = empty((U, 8), Float)
        B = empty((8, R), Float)
        theta = empty(self.volShape, Float)

        # build B matrix, always stays the same
        B[0,:] = b1[0]
        B[1,:] = (arange(R)-R/2)*b2
        B[2,:] = (arange(R)-R/2)*b7[0]
        B[3,:] = b3
        B[4,:] = b8[0]
        B[5,:] = b4
        B[6,:] = (arange(R)-R/2)*b5[0]
        B[7,:] = b6[0]
        


##         B[0] = (arange(R)-R/2)*b7
##         B[1] = (arange(R)-R/2)*b5
##         B[2,:] = b1[0]
##         B[3,:] = b8[0]
##         B[4,:] = b6[0]
        # u_line & zigzag define how the correction changes per PE line
        u_line = arange(U)
        zigzag = checkerline(U)
        # build A matrix, changes slightly as s varies
        A[:,0] = zigzag
        A[:,1] = zigzag
        A[:,2] = zigzag
        A[:,5] = u_line
        A[:,6] = u_line
##         A[:,0] = zigzag
##         A[:,1] = u_line
##         A[:,2] = zigzag
        for s in range(S):
            A[:,3] = s*zigzag
            A[:,4] = s*zigzag
            A[:,7] = s*u_line
            theta[s] = matrixmultiply(A,B)
        return theta

    def run(self, image):
        if not image.ref_data or len(image.ref_vols) < 2:
            self.log("Not enough reference volumes, quitting.")
            return

        self.volShape = image.data.shape[1:]
        
        #shift_columns_left(image.ref_data[0])
        inv_ref0 = ifft(image.ref_data[0])
        shift_columns_left(image.ref_data[1])
        #shift_columns_right(image.ref_data[1])
        inv_ref1 = ifft(image.ref_data[1])
        inv_ref = inv_ref0*conjugate(inv_ref1)
        
        n_slice, n_pe, n_fe = self.refShape = inv_ref0.shape

        lin_pix = int(round(self.lin_radius/image.xsize))
        (self.lin1, self.lin2) = (lin_pix > n_fe/2) and (0,n_fe) or \
                                 ((n_fe/2-lin_pix), (n_fe/2+lin_pix))
        self.lin_fe = self.lin2-self.lin1
        
        # comes back smaller! read direction goes from lin1:lin2
        phs_vol = unwrap_ref_volume(angle(inv_ref), self.lin1, self.lin2)
        
        s_mask = zeros(n_slice) # this will be 4 most "linear" slices
        r_mask = ones((self.lin_fe)) #
        u_mask = ones((n_pe))  # 
        u_mask[0] = 0
        u_mask[46:] = 0
        res = zeros((n_slice,), Float)
        for s in range(n_slice):
            for row in phs_vol[s]:
                (_, _, r) = linReg(arange(self.lin_fe), row)
                res[s] += r
                #plot(row)
            #show()

        # find 4 slices with smallest residual
        sres = sort(res)
        selected = [find(res==c)[0] for c in sres[:4]]
        for c in selected:
            s_mask[c] = 1
        
        self.coefs = (b1,b7,b8,b5,b6) = \
                         self.solve_phase(phs_vol, r_mask, u_mask, s_mask)
        print self.coefs
        Tl = image.T_pe
        delT = 1./image._procpar.sw[0]
        print "Tl = %f; delT = %f"%(Tl,delT,)
        theta_vol = self.correction_volume(Tl, delT)

        for dvol in image.data:
            dvol[:] = apply_phase_correction(dvol, -theta_vol)
