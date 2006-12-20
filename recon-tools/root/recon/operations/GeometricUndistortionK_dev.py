import Numeric as N
from MLab import squeeze
from LinearAlgebra import solve_linear_equations as solve

from recon.operations import Operation, Parameter, verify_scanner_image
from recon.operations.ReadImage import ReadImage as ReadIm
from recon.operations.GaussianSmooth import gaussian_smooth
from recon.util import fft, ifft, fft2d, epi_trajectory

class GeometricUndistortionK_dev (Operation):
    """
    Use a fieldmap to perform geometric distortion correction in k-space.
    Specify:
    @param fmap_file: Name of the field map file
    @param lmbda: Inverse regularization factor
    """

    params=(
        Parameter(name="fmap_file", type="str", default="fieldmap-0",
                  description="Name of the field map file"),
        Parameter(name="lmbda", type="float", default=2.0,
                  description="Inverse regularization factor"),
        Parameter(name="basis", type="str", default="fourier",
                  description="type of basis to use when computing kernel"),
        Parameter(name="xform_target", type="str", default="k-space",
                  description="space to transform data to "\
                  "(k-space or im-space)"),
        
        )

    def fourier_basis(self, N2, N2P, M2):
        df = 2.j*N.pi*N.fromfunction(lambda y,x: x-y, (N2,N2P))/float(M2)
        mv = N.arange(M2)-M2/2
        # outerproduct of df_n, mv is effectively outerproduct(df_n.flat, mv)
        # I want to shape this differently, so that N2xM is the 1st face of
        # the 3d matrix. This way, the N2xM plane of the second exponential
        # gets repeatedly multiplied along the N2P dimension correctly.
        # after this multiplication, switch the dimensions back to
        # (N2,N2P,M) and sum along m--leaving a correct N2xN2P grid

        # returning the magnitude of the basis will be fine
        return (N.ones((N2P,M2)),
                N.swapaxes(N.reshape(N.exp(N.outerproduct(df,mv))/float(M2),
                                 (N2,N2P,M2)),
                         0, 1))

    def fourier_recon(self, coefs):
        return ifft(coefs)
    
    def haar_basis(self, N2, N2P, M2):
        nscales = int(N.floor(N.log(N2)/N.log(2)))
        psi = N.empty((N2P,M2), N.Float)
        psi[0] = N.ones((M2,), N.Float)/N.power(M2,0.5)
        psi[1] = haar_scaling_and_translations(j=0, M=M2)
        for j in range(nscales):
            psi[N.power(2,j):N.power(2,j+1)] = \
                                    haar_scaling_and_translations(j=j, M=M2)
        basis_xform = N.empty((N2P,N2,M2), N.Complex)
        e0 = N.exp(-2.j*N.pi*N.outerproduct(N.arange(N2)-N2/2, N.arange(M2)-M2/2)/M2)
        for scl in range(N2P):
            basis_xform[scl] = psi[scl]*e0
        return psi,basis_xform

    def haar_recon(self, coefs, j=-1, M=None):
        if j < 0:
            M = coefs.shape[-1]
            nscales = int(N.log(M)/N.log(2))
##             if len(coefs) != power(2,nscales):
##                 raise ValueError("not enough coefficients for %d levels"%nscales)
            # first add normalized scaling
            y = N.ones((M,), N.Complex)*(coefs[0]/N.power(M,0.5))
            #y = N.N.zeros((M,), N.complex128)
            for j in range(nscales):
                y += self.haar_recon(coefs[N.power(2,j):N.power(2,j+1)],j=j,M=M)
            return y
        ntrans = N.power(2,j)        
        psi_len = M/ntrans
        psi_trans = haar_scaling_and_translations(j=j, M=M)

        if j==0:
            return N.asarray(coefs)[0] * psi_trans
        else:
            y = N.zeros((M,), N.Complex)
            for k in range(ntrans):
                y += coefs[k]*psi_trans[k]
            return y

    def run(self, image):
        if not verify_scanner_image(self, image):
            return
        basis_func, recon_func = {'fourier': (self.fourier_basis,
                                              self.fourier_recon),
                                  'haar': (self.haar_basis,
                                           self.haar_recon),
                                  }[self.basis]

        fmapIm = ReadIm(**{'filename': self.fmap_file,
                              'format': 'nifti'}).run()

        (nslice, npe, nfe) = image.shape[-3:]
        # make sure that the length of the q1 columns of the fmap
        # are AT LEAST equal to that of the image
        regrid_fac = max(npe, fmapIm.shape[-2])
        fmap = regrid(fmapIm[0], regrid_fac, axis=-2).astype(N.Float)
        chi = regrid(fmapIm[1], regrid_fac, axis=-2)

        # timer stuff:
        import time
        lag = -time.time() + time.time()

        #Q1 = nfe
        #Q2 = N2 = N2P = npe
        #Q1 = nfe
        M2,M1 = fmap.shape[-2:]
        N2 = N2P = npe

        basis_vecs, basis_xform = basis_func(N2,N2P,M2)
        #basis_recon = recon_funcs[self.basis]
        # compute T_n2 vector
        Tl = image.T_pe
        delT = image.delT
        a, b = epi_trajectory(image.nseg, image.sampstyle, N2) 
        T_n2 = (a*delT/2. + b*Tl)
        

        for s in range(nslice):
            # make matrix K[q1;n2,n2p] slice-by-slice
            print "finding distortion kernel Ks for s = %d"%(s,)
            K = N.empty((M1,N2,N2P), N.Complex)
            valid_pts = N.empty((M1,N2P))
            start = time.time()
            e2 = chi[s]*N.exp(1.j*N.reshape(N.outerproduct(T_n2,fmap[s]),(N2,M2,M1)))
            #e2[:] = gaussian_smooth(e2, 3, 3)
            for q1 in range(M1):

                K[q1][:] = N.sum(N.swapaxes(basis_xform*e2[:,:,q1], 0, 1),
                                 axis=-1)

                in_chi = N.nonzero(chi[s,:,q1])

                abs_overlap = N.power(N.take(basis_vecs, in_chi, axis=-1), 2.0)
                valid_pts[q1] = N.where(N.sum(abs_overlap, axis=-1) > 0, 1, 0)

            #### TRY TO SWITCH LOOPS LATER ####
            for dvol in image:
                invdata = ifft(dvol[s])
                #must make this array contiguous in memory along N2
                coefdata = N.zeros((M1,N2), N.Complex)
                for q1 in range(M1):
                    vpts = N.nonzero(valid_pts[q1])
                    if self.basis == "fourier": vpts = N.arange(N2)
                    if N.sum(vpts):
                        # take good pts only into solution
                        N.put(coefdata[q1], vpts,
                              solve_regularized_eqs(N.take(K[q1],vpts,axis=-1),
                                                    invdata[:,q1],
                                                    self.lmbda))
                        # original method was fourier+k-space target, and I
                        # did not do any special recon, but DID
                        # transform along Q1 at the end
                        if self.basis != "fourier" or \
                            self.xform_target != "k-space":
                            dvol[s,:,q1] = recon_func(coefdata[q1])
                    else:
                        dvol[s,:,q1] = 0.j
                
                if self.xform_target == "k-space":
                    if self.basis == "fourier":
                        coefdata = N.transpose(coefdata)
                        dvol[s] = fft(coefdata)
                    else:
                        dvol[s] = fft2d(dvol[s])
                    
            end = time.time()
            #print "time to process %d slice(s): %fs"%(image.nvol,(end-start-lag))
            
def solve_regularized_eqs(A, y, lmbda):
    At = N.conjugate(N.transpose(A))
    A2 = (lmbda**2)*N.identity(At.shape[0]) + N.dot(At, A)
    y2 = N.dot(At, y)
    return solve(A2, y2)

def haar_scaling_and_translations(j=0, M=64, normalized=True):
    if j < 0: return
    scale = N.power(2,j)
    psi_len = M/scale
    w_shape = (M/psi_len,M)
    psi = N.zeros((psi_len,), N.Float)
    psi[:psi_len/2] = N.ones((psi_len/2,))
    psi[psi_len/2:] = -N.ones((psi_len/2,))
    if normalized:
        psi[:] = psi/N.power(psi_len, 0.5)
    w = N.zeros(w_shape, N.Float)
    w[0,:psi_len] = psi
    for k in range(0,M/psi_len):
        w[k,k*psi_len:(k+1)*psi_len] = psi
    return squeeze(w)

def regrid(fmap, P, axis=-1):
    M = float(fmap.shape[-2])
    pset = N.arange(P)*M/P + M/(2*P)
    p_indices = N.floor(pset).astype(N.Int32)
    return N.take(fmap, p_indices, axis)
