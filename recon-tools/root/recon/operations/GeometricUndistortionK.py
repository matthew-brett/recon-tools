from numpy.core import intc
from numpy.linalg import lapack_lite
import numpy as N

from recon.operations import Operation, Parameter, verify_scanner_image
from recon.nifti import readImage
from recon.util import fft, ifft, checkerline

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

    def run(self, image):
        if not verify_scanner_image(self, image):
            return
        try:
            fmapIm = readImage(self.fmap_file)
        except:
            self.log("fieldmap not found: "+self.fmap_file)
            return -1
        (nslice, npe, nfe) = image.shape[-3:]
        # make sure that the length of the q1 columns of the fmap
        # are AT LEAST equal to that of the image
        regrid_fac = max(npe, fmapIm.shape[-2])
        # fmap and chi-mask are swapped to be of shape (Q1,Q2)
        fmap = N.swapaxes(regrid(fmapIm[0], regrid_fac, axis=-2).astype(N.float64), -1, -2)
        chi = N.swapaxes(regrid(fmapIm[1], regrid_fac, axis=-2), -1, -2)
        Q1,Q2 = fmap.shape[-2:]
        
        # compute T_n2 vector
        Tl = image.T_pe
        delT = image.delT

        a, b, n2, _ = image.epi_trajectory()

        K = get_kernel(Q2, Tl, b, n2, fmap, chi)

        for s in range(nslice):
            # dchunk is shaped (nvol, npe, nfe)
            # inverse transform along nfe
            dchunk = ifft(image[:,s,:,:])
            # now shape is (nfe, npe, nvol)
            dchunk = N.swapaxes(dchunk, 0, 2)
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
                dchunk[fe] = N.dot(iK, dchunk[fe])
            dchunk = N.swapaxes(dchunk, 0, 2)
            image[:,s,:,:] = fft(dchunk)

def get_kernel(Q2, Tl, b, n2, fmap, chi):
    # fmap, chi are shaped (Q3,Q1,Q2)
    # kernel is exp(f[q3,q1,n2,q2]) --IFFT-->> exp(f[q3,q1,n2,n2p])
    T_n2 = b*Tl
    q2_ax = N.arange(-Q2/2., Q2/2.)
    zarg = fmap[:,:,None,:] * T_n2[None,None,:,None] - \
           (2*N.pi*n2[:,None]*q2_ax[None,:]/Q2)
    K = N.exp(1.j*zarg)
    del zarg    
    N.multiply(K, chi[:,:,None,:], K)
    K = ifft(K)
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

def regularized_inverse(A, lmbda):
    # I think N.linalg.solve can be sped-up for this special case
    m,n = A.shape
    Ah = N.transpose(N.conj(A))
    A2 = (lmbda**2)*N.identity(n, N.complex128) + N.dot(Ah,A)
    pivots = N.zeros(n, intc)
    results = lapack_lite.zgesv(n, m, A2, n, pivots, A, n, 0)
    return N.transpose(N.conj(A))

def regrid(fmap, P, axis=-1):
    M = float(fmap.shape[-2])
    pset = N.arange(P)*M/P + M/(2*P)
    p_indices = N.floor(pset).astype(N.int32)
    return N.take(fmap, p_indices, axis=axis)
