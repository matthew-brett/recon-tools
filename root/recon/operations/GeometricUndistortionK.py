from numpy.core import intc
from numpy.linalg import lapack_lite
import numpy as np

from recon.operations import Operation, Parameter, verify_scanner_image, \
     ChannelIndependentOperation
from recon.imageio import clean_name
from recon.nifti import readImage
from recon.fftmod import fft1, ifft1

class GeometricUndistortionK (Operation):
    """
    Uses a fieldmap to calculate the geometric distortion kernel for each PE
    line in an image, then applies the inverse operator on k-space data to
    correct for field inhomogeneity distortions.
    """

    params=(
        Parameter(name="fmap_file", type="str", default="fieldmap-0",
            description="""
    Name of the field map file."""),
        Parameter(name="lmbda", type="float", default=8.0,
            description="""
    Inverse regularization factor."""),        
        )

    @ChannelIndependentOperation
    def run(self, image):
        if not verify_scanner_image(self, image):
            return
        fmap_file = clean_name(self.fmap_file)[0]
##         if hasattr(image, 'n_chan'):
##             fmap_file += '.c%02d'%image.chan
        try:
            fmapIm = readImage(fmap_file)
        except:
            self.log("fieldmap not found: "+fmap_file)
            return -1
        (nslice, npe, nfe) = image.shape[-3:]
        # make sure that the length of the q1 columns of the fmap
        # are AT LEAST equal to that of the image
        regrid_fac = max(npe, fmapIm.shape[-2])
        # fmap and chi-mask are swapped to be of shape (Q1,Q2)
        fmap = np.swapaxes(regrid_bilinear(fmapIm[0], regrid_fac, axis=-2).astype(np.float64), -1, -2)
        chi = np.swapaxes(regrid_bilinear(fmapIm[1], regrid_fac, axis=-2), -1, -2)
        Q1,Q2 = fmap.shape[-2:]
        
        # compute T_n2 vector
        Tl = image.T_pe
        delT = image.delT

        a, b, n2, _ = image.epi_trajectory()

        K = get_kernel(Q2, Tl, b, n2, fmap, chi)

        for s in range(nslice):
            # dchunk is shaped (nvol, npe, nfe)
            # inverse transform along nfe (can't do in-place)
            dchunk = ifft1(image[:,s,:,:])
            # now shape is (nfe, npe, nvol)
            dchunk = np.swapaxes(dchunk, 0, 2)
            for fe in range(nfe):
                # want to solve Kx = y for x
                # K is (npe,npe), and y is (npe,nvol)
                #
                # There seems to be a trade-off here as nvol changes...
                # Doing this in two steps is faster for large nvol; I think
                # it takes advantage of the faster BLAS matrix-product in dot
                # as opposed to LAPACK's linear solver. For smaller values
                # of nvol, the overhead seems to outweigh the benefit.
                iK = regularized_inverse(K[s,fe], self.lmbda)
                dchunk[fe] = np.dot(iK, dchunk[fe])
            dchunk = np.swapaxes(dchunk, 0, 2)
            # fft x back to kx, can do inplace here
            fft1(dchunk, inplace=True)
            image[:,s,:,:] = dchunk

def get_kernel(Q2, Tl, b, n2, fmap, chi):
    # fmap, chi are shaped (Q3,Q1,Q2)
    # kernel is exp(f[q3,q1,n2,q2]) --IFFT-->> exp(f[q3,q1,n2,n2p])
    q2_ax = np.arange(-Q2/2., Q2/2., dtype=fmap.dtype)
    T_n2 = fmap.dtype.type(b*Tl)
    n2 = fmap.dtype.type(n2)
    pi = fmap.dtype.type(np.pi)
    zarg = fmap[:,:,None,:] * T_n2[None,None,:,None] - \
           (2*pi*n2[:,None]*q2_ax[None,:]/Q2)
    print zarg.dtype
    K = np.empty(zarg.shape, zarg.dtype.char.upper())
    np.cos(zarg, K.real)
    np.sin(zarg, K.imag)
    del zarg
    np.multiply(K, chi[:,:,None,:], K)
    ifft1(K, inplace=True)
    # a little hacky here.. if k-space was sampled asymmetrically,
    # then we only want to keep as many n2p points as there are n2 points
    # the points should be last len(n2) points in the N2P dimension
    # To find out if the image was sampled asymmetrically, check
    # that n2 has as many points >=0 as it does < 0

    n_pe_really = 2*(n2>=0).sum()
    if n2.shape[0] < n_pe_really:
        subslice = [slice(None)]*len(K.shape)
        n2pts = len(n2)
        subslice[-1] = slice(-n2pts, None, None)
        Ktrunc = K[subslice].copy()
        del K
        return Ktrunc
    return K

def regularized_inverse(A, lm):
    """
    This computes the gereralized inverse of A through a 
    regularized solution: (Ah*A + lm^2*I)*C = Ah*I
    
    use CBLAS to get the matrix product of (Ah*A + lm^2*I) in A2,
    and then use LAPACK to solve for C. 
    
    Note: in column-major...
    A is conj(Ah) 
    A2 is conj(Ah*A + lm^2*I) -- since A2 is hermitian symmetric
    
    If conj(AhA + (lm^2)I)*C = conj(Ah), then conj(C) (the conjugate of
    the LAPACK solution, which is in col-major) is the desired solution
    in column-major. So the final answer in row-major is the hermitian
    transpose of C.
    """

    m,n = A.shape
    # Ah is NxM 
    Ah = np.empty((n,m), A.dtype)
    Ah[:] = A.transpose().conjugate()

    # A2 is NxN
    A2 = np.dot(Ah,A)
    # add lm**2 to the diagonal (ie, A2 + (lm**2)*I)
    A2.flat[0:n*n:n+1] += (lm*lm)

    pivots = np.zeros(n, intc)

    # A viewed as column major is considered to be NxNRHS (NxM)...
    # the solution will be NxNRHS too, and is stored in A
    results = lapack_lite.zgesv(n, m, A2, n, pivots, A, n, 0)
    
    # put conjugate solution into row-major (MxN) by hermitian transpose
    Ah[:] = A.transpose().conjugate()
    return Ah

def regrid(fmap, P, axis=-1):
    M = float(fmap.shape[axis])
    pset = np.arange(P)*M/P + M/(2*P)
    p_indices = np.floor(pset).astype(np.int32)
    return np.take(fmap, p_indices, axis=axis)

def regrid_bilinear(fmap, P, axis=-1):
    M = fmap.shape[axis]
    psamp = np.arange(P, dtype=fmap.dtype)*M/P
    upper_bounds = np.floor(psamp+1).astype(np.int32)
    lower_bounds = np.floor(psamp).astype(np.int32)
    # fix upper_bounds so as not to extend outside of array
    np.putmask(upper_bounds, upper_bounds>(M-1), 0)
    
    p_dist = (psamp - lower_bounds).astype(fmap.dtype)
    dvshape = [1]*len(fmap.shape)
    dvshape[axis] = P
    p_dist.shape = tuple(dvshape)
    
    slicer1 = [slice(None)] * len(fmap.shape)
    slicer2 = [slice(None)] * len(fmap.shape)
    slicer1[axis] = lower_bounds
    slicer2[axis] = upper_bounds

    return (1.0-p_dist)*fmap[slicer1] + p_dist*fmap[slicer2]
