from imaging.operations import Operation, Parameter
from imaging.operations.ReadImage import ReadImage as ReadIm
from imaging.util import fft, ifft
from pylab import pi, arange, exp, zeros, ones, empty, inverse, Complex, \
     find, dot, asum, take, matrixmultiply, Complex32, fromfunction, outerproduct, reshape, svd, transpose, conjugate, identity, Float, swapaxes
import LinearAlgebra as LA
from LinearAlgebra import solve_linear_equations as solve
from _kcalc import kCalc

class GeometricUndistortionK (Operation):
    "Use a fieldmap to perform geometric distortion correction in k-space"

    params=(
        Parameter(name="fmap_file", type="str", default="fieldmap-0",
                  description="Name of the field map file"),
        Parameter(name="mask_file", type="str", default="volmask-0",
                  description="Name of the volume mask file")
        )

    def run(self, image):
        
        fmapIm = ReadIm(**{'filename': self.fmap_file,
                              'format': 'analyze'}).run()
        bmaskIm = ReadIm(**{'filename':self.mask_file,
                               'format': 'analyze'}).run()

        fmap = fmapIm.data.astype(Float)
        bmask = bmaskIm.data

        (nvol, nslice, npe, nfe) = image.data.shape


        # timer stuff:
        import time
        lag = -time.time() + time.time()
        
        Tl = image.T_pe
        Q1 = nfe
        Q2 = N2 = N2P = npe
        df_n = 2.j*pi*fromfunction(lambda y,x: x-y, (N2,N2P))/float(N2)
        n2v = 1.j*(arange(N2)-N2/2)*Tl
        q2v = arange(Q2)-Q2/2
        # outerproduct of df_n, q2v is effectively outerproduct(df_n.flat, q2v)
        # I want to shape this differently, so that N2xQ2 is the 1st face of
        # the 3d matrix. This way, the N2xQ2 plane of the second exponential
        # gets repeatedly multiplied along the N2P dimension correctly.
        # after this multiplication, switch the dimensions back to
        # (N2,N2P,Q2) and sum along q2--leaving a correct N2xN2P grid
        e1 = swapaxes(reshape(exp(outerproduct(df_n,q2v)), (N2,N2P,Q2)), 0, 1)
        for s in range(nslice):
            # make matrix K[q1;n2,n2p] slice-by-slice
            print "finding Ks for s = %d"%(s,)
##             K = kCalc(e1, n2v, fmap[s], bmask[s])
            K = empty((Q1,N2,N2P), Complex)
            start = time.time()
            for q1 in range(Q1):
                chi = bmask[s,:,q1]
                e2 = exp(outerproduct(n2v,fmap[s,:,q1]*chi))
                K[q1][:] = solve_regularized_eqs(
                    asum(chi*swapaxes(e1*e2,0,1), axis=-1)/float(Q2),
                    identity(N2, Complex), 2.0)
                
                #K[q1] = asum(chi*swapaxes(e1*e2,0,1), axis=-1)/float(Q2)

            for dvol in image.data:
                invdata = ifft(dvol[s])
                corrdata = empty((N2,Q1), Complex)
                for q1 in range(Q1):
                    corrdata[:,q1] = dot(K[q1],invdata[:,q1])
                    #corrdata[:,q1] = solve_regularized_eqs(K[q1],invdata[:,q1],2.0)
                dvol[s][:] = fft(corrdata).astype(Complex32)
            end = time.time()
            print "time to process %d slice(s): %fs"%(image.nvol,(end-start-lag))
            
def solve_regularized_eqs(A, y, lmda):
    At = conjugate(transpose(A))
    A2 = (lmda**2)*identity(A.shape[0]) + dot(At, A)
    y2 = dot(At, y)
    return solve(A2, y2)
