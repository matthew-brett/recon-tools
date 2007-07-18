"Applies a Balanced Phase Correction on data sets with two reference volumes"

import numpy as N
import os
from recon.operations import Operation, verify_scanner_image
from recon.operations.ReorderSlices import ReorderSlices
from recon.util import ifft, apply_phase_correction, linReg, checkerline, \
     unwrap_ref_volume, reverse, maskbyfit, polyfit, bivariate_fit
from recon.imageio import readImage

class BalPhaseCorrection (Operation):
    """
    Balanced Phase Correction attempts to reduce N/2 ghosting and other
    systematic phase errors by fitting referrence scan data to a system
    model. This can only be run on special balanced reference scan data.
    """
    

    def run(self, image):
        
        if not verify_scanner_image(self, image):
            return -1
        
        if not hasattr(image, "ref_data") or image.ref_data.shape[0] < 2:
            self.log("Not enough reference volumes, quitting.")
            return -1

        self.volShape = image.shape[-3:]
        inv_ref0 = ifft(image.ref_data[0])
        inv_ref1 = ifft(reverse(image.ref_data[1], axis=-1))

        inv_ref = inv_ref0*N.conjugate(inv_ref1)
        
        n_slice, n_pe, n_fe = self.refShape = inv_ref0.shape

        # let's hardwire this currently??
        (self.lin1,self.lin2) = (0, n_fe)
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

        sigma = N.empty(phs_vol.shape, N.float64)
        #duplicate variance wrt to mu-ev/od over mu for convenience
        sigma[:,0::2,:] = N.power(N.std(phs_vol[:,0::2,:], axis=-2), 2.0)[:,None,:]
        sigma[:,1::2,:] = N.power(N.std(phs_vol[:,1::2,:], axis=-2), 2.0)[:,None,:]

        q1_mask = N.ones((n_slice, n_pe, self.lin_fe))

        # get slice positions (in order) so we can throw out the ones
        # too close to the backplane of the headcoil
        acq_order = image.acq_order
        s_ind = N.concatenate([N.nonzero(acq_order==s)[0] for s in range(n_slice)])
        pss = N.take(image.slice_positions, s_ind)
        bad_slices = (pss < -25.0)
        if bad_slices.any():
            last_good_slice = (pss < -25.0).nonzero()[0][0]
        else:
            last_good_slice = n_slice
        q1_mask[last_good_slice:] = 0.0
        maskbyfit(phs_vol[:last_good_slice],
                  sigma[:last_good_slice], 1.25, 1.25,
                  q1_mask[:last_good_slice])
        
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

##         # planar solution 2
##         q1_mask[:,0,:] = 0
##         q1_mask[:,1,:] = 0
##         print q1_mask.sum(), phs_vol.sum()
##         print r_line_chop
##         coefs = solve_phase2(phs_vol, q1_mask, r_line_chop, s_line)
##         b1,b2,a1,a3,a4 = coefs[:]
##         print coefs
        
##         A = N.empty((self.refShape[-2], 3), N.float64)
##         B = N.empty((3, self.refShape[-1]), N.float64)
##         zigzag = checkerline(self.refShape[-2])
##         B[0,:] = r_line*a1
##         B[1,:] = a3
##         B[2,:] = a4
##         A[:,0] = zigzag
##         A[:,2] = zigzag
##         for s in range(self.refShape[0]):
##             A[:,1] = s*zigzag
##             theta[s] = N.dot(A,B)

##         print theta.sum()
            
##         # planar solution 3
##         q1_mask[:,0,:] = 0
##         q1_mask[:,1,:] = 0
##         A = N.empty((n_pe, 2), N.float64)
##         B = N.empty((2, n_fe), N.float64)
##         zigzag = checkerline(n_pe)
##         for s in range(phs_vol.shape[0]):
##             coefs = solve_phase3(phs_vol[s], q1_mask[s], r_line_chop)
##             a1,a4 = coefs[2:]
##             B[0,:] = r_line*a1
##             B[1,:] = a4
##             A[:,0] = zigzag
##             A[:,1] = zigzag
##             theta[s] = N.dot(A,B)
        
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


##         import matplotlib.axes3d as p3
##         import pylab as P
##         from matplotlib.colors import colorConverter
##         colors = ['b','g','r','c','y','m','k']
##         Rx,Sx = P.meshgrid(r_line_chop, s_line)
##         red = colorConverter.to_rgba_list('r')
##         mid = (self.lin2-self.lin1)/3

##         evn_diff_mu = N.diff(phs_vol[:,::2,mid],axis=-1)
##         odd_diff_mu = N.diff(phs_vol[:,1::2,mid],axis=-1)

##         evn_diff_fe = N.diff(phs_vol[:,::2,:],axis=-1)
##         odd_diff_fe = N.diff(phs_vol[:,1::2,:],axis=-1)

##         evn_diff_fe2 = phs_vol[:,::2,2:] - phs_vol[:,::2,:-2]
##         odd_diff_fe2 = phs_vol[:,1::2,2:] - phs_vol[:,1::2,:-2]

##         (_,evn_slopes,_) = linReg(phs_vol[:,::2,:])
##         (_,odd_slopes,_) = linReg(phs_vol[:,1::2,:])

## ##         bins = N.linspace(min(evn_diff_mu.min(),odd_diff_mu.min()),
## ##                           max(evn_diff_mu.max(),odd_diff_mu.max()),20)
##         bins = N.linspace(min(evn_slopes.min(),odd_slopes.min()),
##                           max(evn_slopes.max(),odd_slopes.max()),20)
        
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

##         for u in range(2,n_pe,11):
##             fig = P.figure()
##             ax = p3.Axes3D(fig)
##             ax.hold(True)
##             ax.plot_wireframe(Rx[:-1],Sx[:-1],0.5*phs_vol[:-1,u,:])
##             ax.plot_wireframe(Rx,Sx,theta[:,u,self.lin1:self.lin2], colors=red)
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

        phase = N.exp(-1.j*theta)
        from recon.tools import Recon
        if Recon._FAST_ARRAY:
            image[:] = apply_phase_correction(image[:], phase)
        else:
            for dvol in image:
                dvol[:] = apply_phase_correction(dvol[:], phase)
        
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

def solve_phase2(pvol, surf_mask, r_line, s_line):
    (B1,B2,A1,A3,A4) = range(5)
    nrows = surf_mask.sum()
    #surf_mask = N.reshape(surf_mask, (S*U, R))
    A = N.empty((nrows, 5), N.float64)
    P = N.empty((nrows, 1), N.float64)
    row_start, row_end = 0, 0
    b = checkerline(pvol.shape[-2])
    for s in range(pvol.shape[0]):
        for u in range(pvol.shape[1]):
            nrows = surf_mask[s,u].sum()
            if not nrows:
                continue
            r_ind = surf_mask[s,u].nonzero()[0]
            row_start = row_end
            row_end = row_start + nrows
            P[row_start:row_end,0] = pvol[s,u,r_ind]
            A[row_start:row_end,B1] = r_line[r_ind]
            A[row_start:row_end,B2] = 1.
            A[row_start:row_end,A1] = b[u]*(2.*r_line[r_ind])
            A[row_start:row_end,A3] = b[u]*(2.*s)
            A[row_start:row_end,A4] = b[u]*(2.)

    [u,s,vt] = N.linalg.svd(A, full_matrices=0) 
    print "cond # = %f"%(s[0]/s[-1])
    invS = N.diag(1/s)
    invS[-1,-1] = 0.0
    V = N.dot(N.transpose(vt), N.dot(invS, N.dot(N.transpose(u),P)))
    return N.transpose(V)[0]

def solve_phase3(pvol, surf_mask, r_line):
    (B1,B2,A1,A4) = range(4)
    nrows = surf_mask.sum()
    A = N.empty((nrows, 4), N.float64)
    P = N.empty((nrows, 1), N.float64)
    row_start, row_end = 0, 0
    b = checkerline(pvol.shape[-2])
    for u in range(pvol.shape[-2]):
        nrows = surf_mask[u].sum()
        if not nrows:
            continue
        r_ind = surf_mask[u].nonzero()[0]
        row_start = row_end
        row_end = row_start + nrows
        P[row_start:row_end,0] = pvol[u,r_ind]
        A[row_start:row_end,B1] = r_line[r_ind]
        A[row_start:row_end,B2] = 1.
        A[row_start:row_end,A1] = b[u]*(2.*r_line[r_ind])
        A[row_start:row_end,A4] = b[u]*(2.)

    [u,s,vt] = N.linalg.svd(A, full_matrices=0)
    print "cond # = %f"%(s[0]/s[-1])
    V = N.dot(N.transpose(vt), N.dot(N.diag(1/s), N.dot(N.transpose(u),P)))
    return N.transpose(V)[0]


