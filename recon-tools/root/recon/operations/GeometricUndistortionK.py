from recon.operations import Operation, Parameter
from recon.operations.ReadImage import ReadImage as ReadIm
from recon.operations.GaussianSmooth import gaussian_smooth
from recon.util import fft, ifft, epi_trajectory
from pylab import pi, arange, exp, zeros, ones, empty, inverse, Complex, \
     find, dot, asum, take, Complex32, fromfunction, \
     outerproduct, reshape, svd, transpose, conjugate, identity, Float, \
     swapaxes, diff, blackman, sign

from LinearAlgebra import solve_linear_equations as solve

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
                  description="Inverse regularization factor")
        )

    def run(self, image):
        
        fmapIm = ReadIm(**{'filename': self.fmap_file,
                              'format': 'nifti'}).run()

        fmap = fmapIm[0].astype(Float)
        bmask = fmapIm[1]

        (nvol, nslice, npe, nfe) = image.data.shape

        # timer stuff:
        import time
        lag = -time.time() + time.time()
        Tl = image.T_pe
        #Q1 = nfe
        #Q2 = N2 = N2P = npe
        Q1 = nfe
        M = fmap.shape[-2]
        N2 = N2P = npe
        delT = 1./image._procpar.sw[0]
        df_n = 2.j*pi*fromfunction(lambda y,x: x-y, (N2,N2P))/float(M)
        a, b = epi_trajectory(image.nseg, image.petable_name, M)
        n2v = 1.j*(a*delT/2. + b*Tl)
        mv = arange(M)-M/2
        # outerproduct of df_n, mv is effectively outerproduct(df_n.flat, mv)
        # I want to shape this differently, so that N2xM is the 1st face of
        # the 3d matrix. This way, the N2xM plane of the second exponential
        # gets repeatedly multiplied along the N2P dimension correctly.
        # after this multiplication, switch the dimensions back to
        # (N2,N2P,M) and sum along m--leaving a correct N2xN2P grid
        e1 = swapaxes(reshape(exp(outerproduct(df_n,mv)), (N2,N2P,M)), 0, 1)
        for s in range(nslice):
            # make matrix K[q1;n2,n2p] slice-by-slice
            print "finding distortion kernel Ks for s = %d"%(s,)
            K = empty((Q1,N2,N2P), Complex)
            start = time.time()
            e2 = bmask[s]*exp(reshape(outerproduct(n2v,fmap[s]),(N2,M,Q1)))
            e2[:] = gaussian_smooth(e2, 3, 3)
            for q1 in range(Q1):

                K[q1][:] = asum(swapaxes(e1*e2[:,:,q1],0,1), axis=-1)/float(M)

                K[q1][:] = solve_regularized_eqs(K[q1],
                                                 identity(N2,Complex),
                                                 self.lmbda)

            for dvol in image.data:
                invdata = ifft(dvol[s])
                corrdata = empty((N2,Q1), Complex)
                for q1 in range(Q1):
                    corrdata[:,q1] = dot(K[q1],invdata[:,q1])

                dvol[s][:] = fft(corrdata).astype(Complex32)
            end = time.time()
            #print "time to process %d slice(s): %fs"%(image.nvol,(end-start-lag))
            
def solve_regularized_eqs(A, y, lmbda):
    At = conjugate(transpose(A))
    A2 = (lmbda**2)*identity(At.shape[0]) + dot(At, A)
    y2 = dot(At, y)
    return solve(A2, y2)
