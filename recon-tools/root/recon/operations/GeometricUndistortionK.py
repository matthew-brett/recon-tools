import numpy as N

from recon.operations import Operation, Parameter, verify_scanner_image
from recon.operations.ReadImage import ReadImage as ReadIm
from recon.operations.GaussianSmooth import gaussian_smooth
from recon.util import fft, ifft, fft2d

class GeometricUndistortionK (Operation):
    """
    Uses a fieldmap to calculate the geometric distortion kernel for each PE
    line in an image, then applies the inverse operator to k-space data.
    """

    params=(
        Parameter(name="fmap_file", type="str", default="fieldmap-0",
            description="""
    Name of the field map file."""),
        Parameter(name="lmbda", type="float", default=2.0,
            description="""
    Inverse regularization factor."""),
        Parameter(name="basis", type="str", default="fourier",
            description="""
    Type of basis to use when computing kernel: haar or fourier
    (leave on default unless brave)."""),
        Parameter(name="xform_space", type="str", default="kspace",
            description="""
    Space to transform data into (kspace or imspace)"""),
        
        )

    def run(self, image):
        if not verify_scanner_image(self, image):
            return
        basis_func, recon_func = {'fourier': (fourier_basis,
                                              fourier_recon),
                                  'haar': (haar_basis,
                                           haar_recon),
                                  }[self.basis]

        fmapIm = ReadIm(filename=self.fmap_file, format='nifti').run()

        (nslice, npe, nfe) = image.shape[-3:]
        # make sure that the length of the q1 columns of the fmap
        # are AT LEAST equal to that of the image
        regrid_fac = max(npe, fmapIm.shape[-2])
        fmap = regrid(fmapIm[0], regrid_fac, axis=-2).astype(N.float64)
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
        a, b = image.epi_trajectory()
        T_n2 = (a*delT/2. + b*Tl)
        

        for s in range(nslice):
            # make matrix K[q1;n2,n2p] slice-by-slice
            print "finding distortion kernel Ks for s = %d"%(s,)
            K = N.empty((M1,N2,N2P), N.complex128)
            valid_pts = N.empty((M1,N2P))
            start = time.time()
            e2 = chi[s]*N.exp(1.j*N.reshape(N.outer(T_n2,fmap[s]),(N2,M2,M1)))
            #e2[:] = gaussian_smooth(e2, 3, 3)
            for q1 in range(M1):

                K[q1][:] = N.swapaxes(basis_xform*e2[:,:,q1],0,1).sum(axis=-1)

                in_chi = chi[s,:,q1].nonzero()[0]
                valid_pts[q1] = N.take(basis_vecs,in_chi,axis=-1).any(axis=-1)

            valid_pts = valid_pts.nonzero()

            # If we're doing a fourier-based kernel, then it will always
            # be full at every volume (there will be nonzero coefficients
            # for all N2). In this case we can have savings by inverting
            # the Q1 kernels here, outside the by-volume loop.

            if self.basis == "fourier":
                for kern in K:
                    kern[:] = solve_reg_eqs(kern, N.identity(N2,N.complex128),
                                            self.lmbda)
                for dvol in image:
                    invdata = ifft(dvol[s])
                    coefdata = N.zeros((N2,M1), N.complex128)
                    for q1 in range(M1):
                        coefdata[:,q1] = N.dot(K[q1],invdata[:,q1])
                        # if going to img space, transform this q1 column
                        # from n2 -> q2
                        if self.xform_space != "kspace":
                            dvol[s,:,q1] = recon_func(coefdata[:,q1])
                    # if going to k-space, transform q1 -> n1
                    if self.xform_space == "kspace":
                        dvol[s] = fft(coefdata)

            else:
                for dvol in image:
                    invdata = ifft(dvol[s])
                    coefdata = N.zeros((N2,M1), N.complex128)
                    for q1 in range(M1):
                        vpts = valid_pts[1][valid_pts[0]==q1]
                        if vpts.any():
                            # take good pts only into solution
                            coefdata[vpts,q1] = solve_reg_eqs(K[q1][:,vpts],
                                                              invdata[:,q1],
                                                              self.lmbda)
                            # in haar-basis case (only alternative to fourier
                            # as of now), reconstruct the q1 column in q2
                            dvol[s,:,q1] = recon_func(coefdata[:,q1])
                        else:
                            dvol[s,:,q1] = 0.j

                    # if going to kspace, transform (q2,q1) -> (n2,n1)
                    if self.xform_space == "kspace":
                        dvol[s] = fft2d(dvol[s])
                    
            end = time.time()
            print "time to process %d slice(s): %fs"%(image.nvol,(end-start-lag))



def fourier_basis(N2, N2P, M2):
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
            N.swapaxes(N.reshape(N.exp(N.outer(df,mv))/float(M2), (N2,N2P,M2)),
                       0, 1))

def fourier_recon(coefs):
    return ifft(coefs)
    
def haar_basis(N2, N2P, M2):
    # this method needs to be generalized for when pe0 != N2/2 
    nscales = int(N.floor(N.log(N2)/N.log(2)))
    psi = N.empty((N2P,M2), N.float)
    psi[0] = N.ones((M2,), N.float)/N.power(M2,0.5)
    psi[1] = haar_scaling_and_translations(j=0, M=M2)
    for j in range(nscales):
        psi[N.power(2,j):N.power(2,j+1)] = \
                                    haar_scaling_and_translations(j=j, M=M2)
    basis_xform = N.empty((N2P,N2,M2), N.complex128)
    # - N2/2 needs to be fixed in case pe0 is not -N2/2 (asym sampling)
    e0 = N.exp(-2.j*N.pi*N.outer(N.arange(N2)-N2/2, N.arange(M2)-M2/2)/M2)
    for scl in range(N2P):
        basis_xform[scl] = psi[scl]*e0
    return psi,basis_xform

def haar_recon(coefs, j=-1, M=None):
    if j < 0:
        M = coefs.shape[-1]
        nscales = int(N.log(M)/N.log(2))
        # first add normalized scaling
        y = N.ones((M,), N.complex128)*(coefs[0]/N.power(M,0.5))
        #y = N.N.zeros((M,), N.complex128)
        for j in range(nscales):
            y += haar_recon(coefs[N.power(2,j):N.power(2,j+1)],j=j,M=M)
        return y
    ntrans = N.power(2,j)        
    psi_len = M/ntrans
    psi_trans = haar_scaling_and_translations(j=j, M=M)
    if j==0:
        return N.asarray(coefs)[0] * psi_trans
    else:
        y = N.zeros((M,), N.complex128)
        for k in range(ntrans):
            y += coefs[k]*psi_trans[k]
        return y
    
def solve_reg_eqs(A, y, lmbda):
    At = N.conjugate(N.transpose(A))
    A2 = (lmbda**2)*N.identity(At.shape[0]) + N.dot(At, A)
    y2 = N.dot(At, y)
    return N.linalg.solve(A2, y2)

def haar_scaling_and_translations(j=0, M=64, normalized=True):
    if j < 0: return
    scale = N.power(2,j)
    psi_len = M/scale
    w_shape = (M/psi_len,M)
    psi = N.zeros((psi_len,), N.float64)
    psi[:psi_len/2] = N.ones((psi_len/2,))
    psi[psi_len/2:] = -N.ones((psi_len/2,))
    if normalized:
        psi[:] = psi/N.power(psi_len, 0.5)
    w = N.zeros(w_shape, N.float64)
    w[0,:psi_len] = psi
    for k in range(0,M/psi_len):
        w[k,k*psi_len:(k+1)*psi_len] = psi
    return N.squeeze(w)

def regrid(fmap, P, axis=-1):
    M = float(fmap.shape[-2])
    pset = N.arange(P)*M/P + M/(2*P)
    p_indices = N.floor(pset).astype(N.int32)
    return N.take(fmap, p_indices, axis=axis)
