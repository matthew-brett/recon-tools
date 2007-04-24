"Applies a Balanced Phase Correction on data sets with two reference volumes"

import numpy as N
import os
from recon.operations import Operation, Parameter, verify_scanner_image
from recon.operations.ReorderSlices import ReorderSlices
from recon.util import ifft, apply_phase_correction, linReg, checkerline, \
     unwrap_ref_volume, reverse, polyfit, bivariate_fit
from recon.imageio import readImage

class BalPhaseCorrection_new (Operation):
    """
    Apply a phase correction with solved-for parameters based on a
    "balanced" pair of reference scans.
    """

    params = (
        Parameter(name="lin_region", type="tuple", default=(-32,32),
                  description="""
    Radius of the region of greatest linearity within the magnetic gradient
    field, in mm (normally 70-80mm)."""),
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
        #inv_ref1 = ifft(image.ref_data[1])
        inv_ref = inv_ref0*N.conjugate(inv_ref1)
        
        n_slice, n_pe, n_fe = self.refShape = inv_ref0.shape

        #lin_pix = int(round(self.lin_radius/image.dFE))

##         (self.lin1, self.lin2) = (lin_pix > n_fe/2) and (0,n_fe) or \
##                                  ((n_fe/2-lin_pix), (n_fe/2+lin_pix))
        (self.lin1,self.lin2) = (n_fe/2 + self.lin_region[0],
                                 n_fe/2 + self.lin_region[1])
        self.lin_fe = self.lin2-self.lin1
        self.alpha, self.beta = image.epi_trajectory()
        # comes back smaller! read direction goes from lin1:lin2
        phs_vol = unwrap_ref_volume(inv_ref, self.lin1, self.lin2)
        
        s_mask = N.zeros(n_slice) # this will be 4 most "linear" slices
        r_mask = N.ones((self.lin_fe)) #
        u_mask = N.ones((n_pe))  # 
        #u_mask[:2] = 0
        #u_mask[50:] = 0

        import pylab as P
        (_,_,res) = linReg(phs_vol, axis=-1)
        res = res.sum(axis=-1)
        #print res
        # find 4 slices with smallest residual
        sres = N.sort(res)
        selected = [N.nonzero(res==c)[0] for c in sres[:4]]
##         for c in selected:
##             s_mask[c] = 1


        sigma = N.empty(phs_vol.shape, N.float64)
        #duplicate variance wrt to mu-ev/od over mu for convenience
        sigma[:,0::2,:] = N.power(N.std(phs_vol[:,0::2,:], axis=-2), 2.0)[:,None,:]
        sigma[:,1::2,:] = N.power(N.std(phs_vol[:,1::2,:], axis=-2), 2.0)[:,None,:]

        q1_mask = N.ones((n_slice, n_pe, self.lin_fe))

        for s in range(n_slice):
            for r in range(n_pe):
                q1_mask[s,r] = maskbyfit(phs_vol[s,r], sigma[s,r], tol=1.5,
                                         tol_growth=1.25, order=2)

        theta = N.empty(self.refShape, N.float64)
        s_line = N.arange(n_slice)
        r_line = N.arange(n_fe) - n_fe/2
        r_line_chop = N.arange(self.lin_fe) + self.lin1 - n_fe/2.

        B1, B2, B3 = range(3)

        # planar solution
        A = N.empty((n_slice, 3), N.float64)
        B = N.empty((3, n_fe), N.float64)
        A[:,0] = 1.
        A[:,1] = s_line
        A[:,2] = 1.
        for u in range(n_pe):
            #q1_mask[:3,u,:] = 0
            #q1_mask[11:,u,:] = 0
            coefs = solve_phase(0.5*phs_vol[:-1,u,:], q1_mask[:-1,u,:],
                                r_line_chop, s_line[:-1])
            B[0,:] = coefs[B1]*r_line
            B[1,:] = coefs[B2]
            #B[1,:] = 0.0
            B[2,:] = coefs[B3]
            theta[:,u,:] = N.dot(A,B)

##         # one line for the whole surface solution
##         sl = 13
##         for mu in range(n_pe):
##             unmasked = q1_mask[sl,mu].nonzero()[0]
##             (b,m,_) = linReg(0.5*phs_vol[sl,mu,unmasked], X=r_line_chop[unmasked])
##             theta[:,mu,:] = r_line*m + b

##         # bivariate surface solution
##         dim0 = N.arange(n_slice)
##         dim1 = N.arange(n_fe)
##         q1_mask_full = N.zeros((n_slice, n_pe, n_fe))
##         q1_mask_full[...,self.lin1:self.lin2] = q1_mask
##         phs_vol_full = N.zeros((n_slice, n_pe, n_fe))
##         phs_vol_full[...,self.lin1:self.lin2] = phs_vol
##         sigma_full = N.zeros((n_slice, n_pe, n_fe))
##         sigma_full[...,self.lin1:self.lin2] = sigma
##         for u in range(n_pe):
##             A,c = bivariate_fit(0.5*phs_vol_full[:,u,:], dim0, dim1, 2,
##                                 sigma=sigma_full[:,u,:],
##                                 mask=q1_mask_full[:,u,:])
##             theta[:,u,:] = N.reshape(N.dot(A,c), (n_slice, n_fe))

##         # line-by-line solution
##         #A = N.empty((self.lin_fe, 2), N.float64)
##         #Ph = N.empty((self.lin_fe, 1), N.float64)
##         for sl in s_line:
##             for mu in range(n_pe):
##                 unmasked = q1_mask[sl,mu].nonzero()[0]
##                 nr = unmasked.shape[-1]
##                 A = N.empty((nr,2), N.float64)
##                 Ph = N.empty((nr,1), N.float64)
##                 Ph[:,0] = 0.5*phs_vol[sl,mu,unmasked]
##                 A[:,0] = r_line_chop[unmasked]
##                 A[:,1] = 1.
##                 [u,s,vt] = N.linalg.svd(A, full_matrices=0)
##                 V = N.dot( N.transpose(vt), N.dot( N.diag(1/s),
##                                                    N.dot(N.transpose(u), Ph)))
##                 theta[sl,mu] = r_line*V[B1] + V[B2]

##         # line-by-line solution with replacement
##         A = N.empty((self.lin_fe, 2), N.float64)
##         Ph = N.empty((self.lin_fe, 1), N.float64)
##         for sl in s_line:
##             for mu in range(n_pe):
##                 dp = N.diff(phs_vol[sl,mu])
##                 med_diff = N.median(dp)
##                 test = N.zeros(dp.shape[-1])
##                 N.putmask(test, abs(dp) > 6*abs(med_diff), 1)
##                 # if line does not have noisy notches, calculate a fit
##                 # otherwise, use the existing fit
##                 if not test.any() or mu<2:

##                     Ph[:,0] = 0.5*phs_vol[sl,mu]
##                     A[:,0] = r_line_chop
##                     A[:,1] = 1.
##                     [u,s,vt] = N.linalg.svd(A, full_matrices=0)
##                     V = N.dot( N.transpose(vt), N.dot( N.diag(1/s),
##                                                        N.dot(N.transpose(u), Ph)))
##                     theta[sl,mu] = r_line*V[B1] + V[B2]                    
##                 else:
##                     print "sl=%d, mu=%d skipped"%(sl,mu)
##                     print test.nonzero()[0]
##                     theta[sl,mu] = theta[sl,mu-2]


##         # funky average solution
##         A = N.empty((self.lin_fe, 2), N.float64)
##         Ph = N.empty((self.lin_fe, 1), N.float64)
##         for sl in s_line:
##             evn = phs_vol[sl,2:n_pe:2].mean(axis=0)
##             odd = phs_vol[sl,3:n_pe:2].mean(axis=0)
##             Ph[:,0] = 0.5*evn
##             A[:,0] = r_line_chop
##             A[:,1] = 1.
##             [u,s,vt] = N.linalg.svd(A, full_matrices=0)
##             V_evn = N.dot( N.transpose(vt), N.dot( N.diag(1/s),
##                                                    N.dot(N.transpose(u), Ph)))
##             Ph[:,0] = 0.5*odd
##             V_odd = N.dot( N.transpose(vt), N.dot( N.diag(1/s),
##                                                    N.dot(N.transpose(u), Ph)))
##             theta[sl,0::2] = r_line*V_evn[B1] + V_evn[B2]
##             theta[sl,1::2] = r_line*V_odd[B1] + V_odd[B2]
            

##         # polyfit solution
##         self.polyorder = 2
##         for sl in range(n_slice):
##             #r_line_chop = q1_mask[sl].nonzero()[0] - n_fe/2
##             for u in range(n_pe):
##                 poly = polyfit(r_line_chop, 0.5*phs_vol[sl,u],
##                                self.polyorder, sigma=sigma[sl,u])
##                 fitted = 0
##                 for n,p in enumerate(poly):
##                     fitted += N.power(r_line,self.polyorder-n)*p
##                 theta[sl,u] = fitted
##                 #theta[sl,u,r_line_chop] = phs_vol[sl,u]
        
##         # trivial solution
##         theta[:] = 0.5*phs_vol


        import matplotlib.axes3d as p3
        import pylab as P
        from matplotlib.colors import colorConverter
        colors = ['b','g','r','c','y','m','k']
        Rx,Sx = P.meshgrid(r_line_chop, s_line)
        red = colorConverter.to_rgba_list('r')
        mid = (self.lin2-self.lin1)/3

        evn_diff_mu = N.diff(phs_vol[:,::2,mid],axis=-1)
        odd_diff_mu = N.diff(phs_vol[:,1::2,mid],axis=-1)

        evn_diff_fe = N.diff(phs_vol[:,::2,:],axis=-1)
        odd_diff_fe = N.diff(phs_vol[:,1::2,:],axis=-1)

        evn_diff_fe2 = phs_vol[:,::2,2:] - phs_vol[:,::2,:-2]
        odd_diff_fe2 = phs_vol[:,1::2,2:] - phs_vol[:,1::2,:-2]

        (_,evn_slopes,_) = linReg(phs_vol[:,::2,:])
        (_,odd_slopes,_) = linReg(phs_vol[:,1::2,:])

##         bins = N.linspace(min(evn_diff_mu.min(),odd_diff_mu.min()),
##                           max(evn_diff_mu.max(),odd_diff_mu.max()),20)
        bins = N.linspace(min(evn_slopes.min(),odd_slopes.min()),
                          max(evn_slopes.max(),odd_slopes.max()),20)
        
##         for s in range(n_slice):
##         #for s in [16,17,18,19]:
##             #for u in range(0,n_pe,5):
##             for u in range(n_pe):
##                 P.plot(r_line_chop, phs_vol[s,u], colors[u%7])
##                 P.plot(r_line, 2*theta[s,u], colors[u%7]+':')
## ##                 if not u%2:
## ##                     P.plot(r_line_chop[1:], evn_diff_fe[s,u/2], 'b.')
## ##                     P.plot(r_line_chop[2:], evn_diff_fe2[s,u/2], 'go')
## ##                 else:
## ##                     P.plot(r_line_chop[1:], odd_diff_fe[s,u/2], 'r.')
## ##                     P.plot(r_line_chop[2:], odd_diff_fe2[s,u/2], 'go')
## ##             P.hist(evn_diff[s],bins=bins)
## ##             P.hist(odd_diff[s],bins=bins, facecolor='red',alpha=.5)
## ##             P.hist(evn_slopes[s],bins=bins)
## ##             P.hist(odd_slopes[s],bins=bins,facecolor='red',alpha=0.5)
##             P.title("slice %d"%(s,))
##             P.grid(True)
##             P.show()
##         for u in range(2,n_pe,9):
##             fig = P.figure()
##             ax = p3.Axes3D(fig)
##             ax.hold(True)
##             ax.plot_wireframe(Rx[:-1],Sx[:-1],0.5*phs_vol[:-1,u,:],colors=red)
##             ax.plot_wireframe(Rx,Sx,theta[:,u,self.lin1:self.lin2])
##             ax.set_xlabel("read-out")
##             ax.set_ylabel("slice")
##             ax.set_zlabel("phase mu=%d"%u)
##             P.show()

##         for s in [0,1,10,19,36,37]:
##             for row in phs_vol[s]:
##                 P.plot(row)
##             P.plot(phs_vol[s,0], 'b.')
##             P.plot(phs_vol[s,1], 'g.')
##             P.title("slice %d"%s)
##             P.show()
        
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = apply_phase_correction(image[:], -theta)
        else:
            for dvol in image:
                dvol[:] = apply_phase_correction(dvol[:], -theta)



def solve_phase(pvol, surf_mask, r_line, s_line):
    # surface solution, pvol is (Nsl)x(Nro)
    # surf_mask is (Nsl)x(Nro)
    # r_line is 1xNro
    # s_line is 1xNsl
    # there is one row for each unmasked point in surf_mask
    (B1,B2,B3) = range(3)
    nrows = surf_mask.sum()
    A = N.empty((nrows, 3), N.float64)
    P = N.empty((nrows, 1), N.float64)
    row_start, row_end = 0, 0
    for s in range(surf_mask.shape[0]):
        
        unmasked = surf_mask[s].nonzero()[0]
        row_start = row_end
        row_end = unmasked.shape[0] + row_start
        P[row_start:row_end,0] = pvol[s,unmasked]
        A[row_start:row_end,B1] = r_line[unmasked]
        A[row_start:row_end,B2] = s_line[s]
        A[row_start:row_end,B3] = 1.
    
    [u,s,vt] = N.linalg.svd(A, full_matrices=0)
    V = N.dot(N.transpose(vt), N.dot(N.diag(1/s), N.dot(N.transpose(u),P)))

        
    return N.transpose(V)[0] 


def composite_mask(M):
    mask = maskbyslope(M)
    foo = maskbylin(M, mask=mask)
    return foo

def maskbyslope(M):
    (_,m,res) = linReg(M)

    # m is a target slope, res is a measure of what to throw out,
    # what to keep

    dM = N.diff(M)
    mask = N.ones(M.shape[-1], N.int32)
    mask[0] = 0
    mask[-1] = 0
    # if there is a diff much larger than m followed by a diff of opposite
    # sign much larger than m, it's a notch.
    for n in range(dM.shape[-1]-1):
        if abs(dM[n]) > 5*m and abs(dM[n+1]) > 5*m and (dM[n]*dM[n+1] < 0):
            mask[n+1] = 0
    return mask

def maskbyfit(M, sigma, tol=None, tol_growth=None,
              mask=None, plot=False, order=1):
    if len(M)==0:
        print "all masked"
        return mask
    if mask is None:
        mask = N.ones(M.shape, N.int32)
    if tol is None:
        tol = 2.0
    if tol_growth is None:
        tol_growth = 1.1
    if order > 2 or order < 1:
        order = 1
    new_mask = N.empty(mask.shape)
    new_mask[:] = mask
    unmasked = new_mask.nonzero()[0]
    xax = N.arange(M.shape[-1])    
    if order == 1:
        (b,m,res) = linReg(M[unmasked], X=unmasked, sigma=sigma[unmasked])
        fitted = xax*m + b
    else:
        poly_c = polyfit(unmasked, M[unmasked], 2, sigma=sigma[unmasked])
        fitted = N.power(xax,2.0)*poly_c[0] + xax*poly_c[1] + poly_c[2]
        res = abs(M[unmasked]-fitted[unmasked]).sum()/unmasked.shape[-1]
    N.putmask(new_mask, abs(M - fitted) > tol*res, 0)
    unmasked = new_mask.nonzero()[0]
    if plot:
        unmasked = new_mask.nonzero()[0]
        (b2,m2,res2) = linReg(M[unmasked], X=xax[unmasked],
                                   sigma=sigma[unmasked])
        P.plot(xax,M)
        P.plot(xax[unmasked], M[unmasked], 'b.')
        P.plot(xax*m + b, 'g')
        P.plot(xax*m2 + b2, 'r')
        P.title("tolerance = %2.4f, avg res = %2.4f, final res = %2.4f"%(tol*res, res, res2))
        P.show()

    if new_mask.all():
        #print "limiting case"
        #print unmasked, new_mask
        return new_mask
    else:
        new_mask[unmasked] = maskbyfit(M[unmasked], sigma[unmasked],
                                       mask=mask[unmasked], order=order,
                                       plot=plot, tol=tol*tol_growth,
                                       tol_growth=tol_growth)
    return new_mask
