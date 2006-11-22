from recon.operations import Operation, Parameter
from recon.operations.ReadImage import ReadImage as ReadIm
from recon.operations.GaussianSmooth import gaussian_smooth
from recon.util import fft, ifft, epi_trajectory, checkerboard
from pylab import pi, arange, exp, zeros, ones, empty, inverse, Complex, \
     nonzero, dot, asum, take, Complex32, fromfunction, squeeze, asarray, \
     outerproduct, reshape, svd, transpose, conjugate, identity, Float, \
     swapaxes, diff, blackman, sign, power, log, floor, where, put

from LinearAlgebra import solve_linear_equations as solve
from FFT import fft2d as _fft2d

class GeometricUndistortionK (Operation):
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
        df = 2.j*pi*fromfunction(lambda y,x: x-y, (N2,N2P))/float(M2)
        mv = arange(M2)-M2/2
        # outerproduct of df_n, mv is effectively outerproduct(df_n.flat, mv)
        # I want to shape this differently, so that N2xM is the 1st face of
        # the 3d matrix. This way, the N2xM plane of the second exponential
        # gets repeatedly multiplied along the N2P dimension correctly.
        # after this multiplication, switch the dimensions back to
        # (N2,N2P,M) and sum along m--leaving a correct N2xN2P grid

        # returning the magnitude of the basis will be fine
        return (ones((N2P,M2)),
                swapaxes(reshape(exp(outerproduct(df,mv))/float(M2),
                                 (N2,N2P,M2)),
                         0, 1))

    def fourier_recon(self, coefs):
        return ifft(coefs)
    
    def haar_basis(self, N2, N2P, M2):
        nscales = int(floor(log(N2)/log(2)))
        psi = empty((N2P,M2), Float)
        psi[0] = ones((M2,), Float)/power(M2,0.5)
        psi[1] = haar_scaling_and_translations(j=0, M=M2)
        for j in range(nscales):
            psi[power(2,j):power(2,j+1)] = \
                                    haar_scaling_and_translations(j=j, M=M2)
        basis_xform = empty((N2P,N2,M2), Complex)
        e0 = exp(-2.j*pi*outerproduct(arange(N2)-N2/2, arange(M2)-M2/2)/M2)
        for scl in range(N2P):
            basis_xform[scl] = psi[scl]*e0
        return psi,basis_xform

    def haar_recon(self, coefs, j=-1, M=None):
        if j < 0:
            M = coefs.shape[-1]
            nscales = int(log(M)/log(2))
##             if len(coefs) != power(2,nscales):
##                 raise ValueError("not enough coefficients for %d levels"%nscales)
            # first add normalized scaling
            y = ones((M,), Float)*(coefs[0]/power(M,0.5))
            #y = N.zeros((M,), N.complex128)
            for j in range(nscales):
                y += self.haar_recon(coefs[power(2,j):power(2,j+1)],j=j,M=M)
            return y
        
        psi_len = M/power(2,j)
        psi_trans = haar_scaling_and_translations(j=j, M=M)
        ntrans = power(2,j)

        if j==0:
            return asarray(coefs)[0] * psi_trans
        else:
            y = zeros((M,), Complex)
            for k in range(ntrans):
                y += coefs[k]*psi_trans[k]
            return y

    def run(self, image):
        
        basis_func, recon_func = {'fourier': (self.fourier_basis,
                                              self.fourier_recon),
                                  'haar': (self.haar_basis,
                                           self.haar_recon),
                                  }[self.basis]

        fmapIm = ReadIm(**{'filename': self.fmap_file,
                              'format': 'nifti'}).run()

        fmap = fmapIm[0].astype(Float)
        chi = fmapIm[1]

        (nvol, nslice, npe, nfe) = image.data.shape

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
        delT = 1./image._procpar.sw[0]
        a, b = epi_trajectory(image.nseg, image.petable_name, N2) 
        T_n2 = (a*delT/2. + b*Tl)
        

        for s in range(nslice):
            # make matrix K[q1;n2,n2p] slice-by-slice
            print "finding distortion kernel Ks for s = %d"%(s,)
            K = empty((M1,N2,N2P), Complex)
            valid_pts = empty((M1,N2P))
            start = time.time()
            e2 = chi[s]*exp(1.j*reshape(outerproduct(T_n2,fmap[s]),(N2,M2,M1)))
            #e2[:] = gaussian_smooth(e2, 3, 3)
            for q1 in range(M1):

                K[q1][:] = asum(swapaxes(basis_xform*e2[:,:,q1], 0, 1),
                                axis=-1)#/float(M2)

                in_chi = nonzero(chi[s,:,q1])
                
                valid_pts[q1] = where(asum(power(take(basis_vecs,
                                                      in_chi,axis=-1), 2.0),
                                           axis=-1) > 0, 1, 0)

            #### TRY TO SWITCH LOOPS LATER ####
            for dvol in image.data:
                invdata = ifft(dvol[s])
                #must make this array contiguous in memory along N2
                coefdata = zeros((M1,N2), Complex)
                for q1 in range(M1):
                    vpts = nonzero(valid_pts[q1])
                    if self.basis == "fourier": vpts = arange(N2)
                    if asum(vpts):
                        # take good pts only into solution
                        put(coefdata[q1], vpts,
                            solve_regularized_eqs(take(K[q1],vpts,axis=-1),
                                                  invdata[:,q1],
                                                  self.lmbda))
                        # original method was fourier+k-space target, and I
                        # did not do any special recon, but DID
                        # transform along Q1 at the end
                        if self.basis != "fourier" or \
                            self.xform_target != "k-space":
                            dvol[s][:,q1] = recon_func(coefdata[q1]).astype(Complex32)
                    else:
                        dvol[s][:,q1] = 0.j
                
                if self.xform_target == "k-space":
                    if self.basis == "fourier":
                        coefdata = transpose(coefdata)
                        dvol[s][:] = fft(coefdata).astype(Complex32)
                    else:
                        dvol[s][:] = fft2d(dvol[s]).astype(Complex32)
                    
            end = time.time()
            #print "time to process %d slice(s): %fs"%(image.nvol,(end-start-lag))
            
def solve_regularized_eqs(A, y, lmbda):
    At = conjugate(transpose(A))
    A2 = (lmbda**2)*identity(At.shape[0]) + dot(At, A)
    y2 = dot(At, y)
    return solve(A2, y2)

def haar_scaling_and_translations(j=0, M=64, normalized=True):
    if j < 0: return
    scale = power(2,j)
    psi_len = M/scale
    w_shape = (M/psi_len,M)
    psi = zeros((psi_len,), Float)
    psi[:psi_len/2] = ones((psi_len/2,))
    psi[psi_len/2:] = -ones((psi_len/2,))
    if normalized:
        psi[:] = psi/power(psi_len, 0.5)
    w = zeros(w_shape, Float)
    w[0,:psi_len] = psi
    for k in range(0,M/psi_len):
        w[k,k*psi_len:(k+1)*psi_len] = psi
    return squeeze(w)

def fft2d(a):
    chk = checkerboard(*a.shape[-2:])
    return chk*_fft2d(a*chk)
