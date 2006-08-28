from imaging.operations import Operation, Parameter
from imaging.operations.ReadImage import ReadImage as ReadIm
from imaging.util import fft, ifft
from pylab import pi, arange, exp, zeros, ones, empty, inverse, Complex, \
     find, dot, asum, take, matrixmultiply, Complex32, fromfunction, outerproduct, reshape, svd, transpose, conjugate, identity, Float, swapaxes, diff, blackman, sign
#import LinearAlgebra as LA
from LinearAlgebra import solve_linear_equations as solve

from scipy.signal import fftconvolve, convolve

def gaussianKernel(fwhm, kernsize=3):
    # assume mu = 0
    from pylab import power, log
    sigma = fwhm*power(8. * log(2), -0.5)
    gax = arange(kernsize) - (kernsize-1)/2.
    gf = exp(-power(gax,2)/(2*sigma**2))
    kern = outerproduct(gf, gf)/(2*pi*sigma**2)
    return kern

def transWin(tw, slope):
    w = blackman(tw*2)
    return slope > 0 and w[:tw] or w[tw:]
    
def smoothBinVect(v):
    #for all up/down transitions, overlay transition window up to 6 points
    chi = v.astype(Float)
    dv = diff(v)
    if sum(abs(dv))==0: return chi
    segs = list(find(dv!=0))
    segs = [segs[0]+1] + diff(segs).tolist() + [len(v)-segs[-1]-1]
    for n, jmp in enumerate(find(dv!=0)+1):
        tw = min(6, min(segs[n], segs[n+1]))
        chi[jmp + tw/2 - tw: jmp + tw/2] = transWin(tw, sign(dv[jmp-1]))
    return chi
        



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
        #Q1 = nfe
        #Q2 = N2 = N2P = npe
        Q1 = nfe
        M = fmap.shape[-2]
        N2 = N2P = npe
        df_n = 2.j*pi*fromfunction(lambda y,x: x-y, (N2,N2P))/float(M)
        n2v = 1.j*(arange(N2)-N2/2)*Tl
        mv = arange(M)-M/2
        # outerproduct of df_n, mv is effectively outerproduct(df_n.flat, mv)
        # I want to shape this differently, so that N2xM is the 1st face of
        # the 3d matrix. This way, the N2xM plane of the second exponential
        # gets repeatedly multiplied along the N2P dimension correctly.
        # after this multiplication, switch the dimensions back to
        # (N2,N2P,M) and sum along m--leaving a correct N2xN2P grid
        e1 = swapaxes(reshape(exp(outerproduct(df_n,mv)), (N2,N2P,M)), 0, 1)
        smooth_kernel = gaussianKernel(3,3)
        from pylab import imshow, show, figure
        for s in range(nslice):
            # make matrix K[q1;n2,n2p] slice-by-slice
            print "finding Ks for s = %d"%(s,)
            K = empty((Q1,N2,N2P), Complex)
            start = time.time()
            e2 = bmask[s]*exp(reshape(outerproduct(n2v,fmap[s]),(N2,M,Q1)))
            for n2 in range(N2):
                e2[n2][:] = convolve(e2[n2],smooth_kernel,mode='same')
            for q1 in range(Q1):

                K[q1][:] = asum(swapaxes(e1*e2[:,:,q1],0,1), axis=-1)/float(M)

                K[q1][:] = solve_regularized_eqs(K[q1],
                                            identity(N2,Complex), 2.0)


##                 K[q1][:] = solve_regularized_eqs(
##                     fftconvolve(smooth_kernel,
##                             asum(chi*swapaxes(e1*e2,0,1), axis=-1)/float(M),
##                             mode='same'),
##                     identity(N2, Complex), 2.0)

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
