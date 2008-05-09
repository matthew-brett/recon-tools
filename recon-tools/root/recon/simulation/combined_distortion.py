import numpy as N, pylab as P
import matplotlib.axes3d as p3
from recon import imageio, util
from recon.simulation import newghosts

def combo_kernel(image, q3, fmap, phi, Tl, filldeg=0, plot=False):
    chi = fmap[1]
    fmap = fmap[0]
    Q3,Q2,Q1 = image.shape[-3:]
    N2,N1 = Q2,Q1

    #n1_ax = N.linspace(0., N1, N1, endpoint=False)
    #n2_ax = N.linspace(0., N2, N2, endpoint=False)
    #q1_ax = N.linspace(0., Q1, Q1, endpoint=False)
    #q2_ax = N.linspace(0., Q2, Q2, endpoint=False)
    n1_ax = N.linspace(-N1/2., N1/2., N1, endpoint=False)
    n2_ax = N.linspace(-N2/2., N2/2., N2, endpoint=False)
    q1_ax = N.linspace(-Q1/2., Q1/2., Q1, endpoint=False)
    q2_ax = N.linspace(-Q2/2., Q2/2., Q2, endpoint=False)

    sl_idx = N.indices((N2,N1), dtype=N.float64)
    # I THINK the EPI scan begins on the THIRD read-out after excitation,
    # so add 3 to the n2 index
    tn2 = (sl_idx[0]+3)*Tl
    tn1 = N.empty((N2,N1))
    t = image.t_n1()
    tn1[0::2] = t
    tn1[1::2] = t[::-1]
    
    tn = (tn2 + tn1)/1e6

    # here is the N1xQ1 exponential
    n1_exp = -2.j*N.pi*n1_ax[:,None]*q1_ax[None,:]/float(Q1)
    # here is the N2xQ2 exponential
    n2_exp = -2.j*N.pi*n2_ax[:,None]*q2_ax[None,:]/float(Q2)

    # cook up phi so it is fully N2xN1xQ1
    phi_up = N.empty((N2,N1,Q1), phi.dtype)
    phi_up[0::2] = phi[0]
    phi_up[1::2] = phi[1]

    if not filldeg:
        theta = fmap[q3]
    else:
        # cook up fmap too with polynomial surface filling in where chi[n] = 0
        # add 1s to the mask at the corneres to bend the solution towards zero
        #r = N.power(sl_idx[0]-N2/2., 2) + N.power(sl_idx[1]-N1/2., 2)
        #constraint_mask = N.where(r >= 38**2, 1, 0)
        constraint_mask = N.where(chi[q3] > 0, 0, 1)
        
        fit_params = util.bivariate_fit(fmap[q3], n1_ax, n2_ax, filldeg,
                                        mask=chi[q3]+constraint_mask)
        fit_surf = N.dot(fit_params[0], fit_params[1])
        fit_surf.shape = (Q2,Q1)
        theta = fmap[q3]*chi[q3] + (1-chi[q3])*fit_surf

    if plot:
        plot3d(theta)
        P.show()
    
    
    # for convenience, put Q2xQ1 in last two dimensions, then FFT them
    # so in the end K will be (N2xN1xN2pxN1p) 
##     zarg = n2_exp[:,None,:,None] + n1_exp[None,:,None,:] + \
##            1.j*tn[:,:,None,None]*fmap[q3,None,None,:,:] + \
##            1.j*phi_up[:,:,None,:]
    zarg = n2_exp[:,None,:,None] + n1_exp[None,:,None,:] + \
           1.j*tn[:,:,None,None]*theta[None,None,:,:] + \
           1.j*phi_up[:,:,None,:]


    ep_n1 = N.exp(2j*N.pi*n1_ax*(Q1/2)/Q1)
    #ep_n1 = N.ones(64)
    ep_n1p = N.exp(-2j*N.pi*n1_ax*(Q1/2)/Q1)
    #ep_n1p = N.ones(64)
    ep_n2 = N.exp(2j*N.pi*n2_ax*(Q2/2)/Q2)
    #ep_n2 = N.ones(64)
    ep_n2p = N.exp(-2j*N.pi*n2_ax*(Q2/2)/Q2)
    #ep_n2p = N.ones(64)


    k = N.exp(zarg)
    del zarg
    print k.sum()
    if not filldeg:
        k = N.multiply(k, chi[q3,None,None,:,:])

    #k = N.multiply(k, ep_n2[:,None,None,None])
    #k = N.multiply(k, ep_n1[None,:,None,None])
    #k = N.multiply(k, ep_n2p[None,None,:,None])
    #k = N.multiply(k, ep_n1p[None,None,None,:])

    print k.sum()
    f = util.ifft2(k, shift=True)
    del k

    #f = N.multiply(f, ep_n2[:,None,None,None])
    #f = N.multiply(f, ep_n1[None,:,None,None])
    #f = N.multiply(f, ep_n2p[None,None,:,None])
    #f = N.multiply(f, ep_n1p[None,None,None,:])
    return f #*ephase

def distort_volume(image, fmap, phi, Tl):
    for q3 in range(image.shape[-3]):
        print "slice",q3, "working"
        k = combo_kernel(image, q3, fmap, phi, Tl)
        apply_kernel(image, k, q3)
    

def apply_kernel(image, k, q3):
    imslice = [slice(None)] * (image.ndim)
    imslice[-3] = q3
    # have dummy dims for N2, N1
    #imslice[-2] = None
    #imslice[-4] = None
    
    d = N.empty(image.shape[-2:], image[:].dtype)
    for n2 in xrange(image.shape[-2]):
        for n1 in xrange(image.shape[-1]):
            print (n2,n1)
            # apply k[n2,n1,n2p,n1p] to s[n2p,n1p] and sum over n1p, n2p
            pt = image[imslice]*k[n2,n1]
            d[n2,n1] = pt.sum()
    image[imslice] = d

def add_sim_ref_data(image, fmap, phi, t_n1, Tl, filldeg=0):
    # image should be CLEAN, so that the DC row can be distorted here and
    # used as ref data
    
    (Q3,Q2,Q1) = image.shape[-3:]
    #f = newghosts.perturbation_kernel(phi)
    rdata = N.empty((Q3,3,Q1), N.complex64)
    mid_row = [slice(None)]*image.ndim
    mid_row[-2] = Q2/2
    if len(mid_row) > 3:
        mid_row[0] = 0
    rdata[:] = (image.data[mid_row])[:,None,:]

    chi = fmap[1]
    fmap = fmap[0]
    Q3,Q2,Q1 = image.shape[-3:]
    N2,N1 = Q2,Q1

    n1_ax = N.linspace(-N1/2., N1/2., N1, endpoint=False)
    u_ax = N.arange(3.)
    q1_ax = N.linspace(-Q1/2., Q1/2., Q1, endpoint=False)
    #q2_ax = N.linspace(-Q2/2., Q2/2., Q2, endpoint=False)
    upr_ax = N.arange(3.)

    sl_idx = N.indices((3,N1), dtype=N.float64)
    tn2 = sl_idx[0]*Tl
    tn1 = N.empty((3,64))
    # simulate neg-pos-neg ref gradients
    tn1[0::2] = t_n1[::-1]
    tn1[1::2] = t_n1

    tn = (tn2+tn1)/1e6

    # cook up phi so it has full (3,N1,Q1) shape
    phi_up = N.empty((3,N1,Q1), phi.dtype)
    phi_up[0::2] = phi[1]
    phi_up[1::2] = phi[0]


    # cook up fmap too with polynomial surface filling in where chi[n] = 0
    if filldeg:
        theta = N.empty((Q3,Q2,Q1), N.float64)
        for q3 in range(Q3):
            fit_params = util.bivariate_fit(fmap[q3],
                                            n1_ax, n1_ax, 3, mask=chi[q3])
            fit_surf = N.dot(fit_params[0], fit_params[1])
            fit_surf.shape = (Q2,Q1)
            theta[q3] = fmap[q3]*chi[q3] + (1-chi[q3])*fit_surf
    else:
        theta = fmap

##     # repeat k=0 phase encode 3 times
##     theta =  N.empty((Q3,3,Q1), N.float64)
##     theta[:] = fmap_filled[:,32,:][:,None,:]

    # want a kernel that's shaped (Q3,3,N1,Q2,N1p), xformed from (Q3,3,N1,Q2,Q1)
    # (no transform over Q2, w/o phase encode it is a projection)
    n1_exp = -2.j*N.pi*n1_ax[:,None]*q1_ax[None,:]/float(Q1)

##     zarg = n1_exp[None,None,:,None,:] + \
##            1.j*tn[None,:,:,None,None]*fmap_filled[:,None,None,:,:] + \
##            1.j*phi_up[None,:,:,None,:]
    zarg = n1_exp[None,None,:,None,:] + \
           1.j*tn[None,:,:,None,None]*theta[:,None,None,:,:] + \
           1.j*phi_up[None,:,:,None,:]

    k = N.exp(zarg)
    del zarg
    if not filldeg:
        N.multiply(k, chi[:,None,None,:,:])
    f = util.ifft1(k)
    del k

    ft_img = util.ifft1(image[:], axis=-2, shift=True)

    rdata[:] = (f * ft_img[:,None,None,:,:]).sum(axis=-1).sum(axis=-1)
    
    # quirkyness-- 0th and 2nd lines are neg gradient, while 1st is pos
    #rdata[:] = (f * rdata[:,:,None,:]).sum(axis=-1)
    #rdata[:,0::2,:] = (f[1,:,:] * rdata[:,0::2,None,:]).sum(axis=-1)
    #rdata[:,1::2,:] = (f[0,:,:] * rdata[:,1::2,None,:]).sum(axis=-1)
    image.ref_data = rdata

def plot3d(m):
    idx = N.indices(m.shape[-2:])
    ax = p3.Axes3D(P.figure())
    ax.plot_wireframe(idx[1], idx[0], m)
    ax.set_zlim(-20,20)

#from recon.operations.GeometricUndistortionK import regularized_inverse
def regularized_inverse(A, lmbda):
    # I think N.linalg.solve can be sped-up for this special case
    m,n = A.shape
    Ah = N.transpose(N.conj(A))
    A2 = (lmbda**2)*N.identity(n, N.complex128) + N.dot(Ah,A)
    pivots = N.zeros(n, N.core.intc)
    results = N.linalg.lapack_lite.zgesv(n, m, A2, n, pivots, A, n, 0)
    return N.conj(A)

def reg_inv2(A, lmbda):
    y = N.identity(A.shape[0], A.dtype)
    At = N.conjugate(N.transpose(A))
    A2 = (lmbda**2)*y + N.dot(At,A)
    y2 = N.dot(At,y)
    return N.linalg.solve(A2,y2)

def unapply_kernel(image, k, q3, lm):
    imslice = [slice(None)] * (image.ndim)
    imslice[-3] = q3

    d = N.empty(image.shape[-2:], image[:].dtype)
    for n2 in xrange(image.shape[-2]):
        for n1 in xrange(image.shape[-1]):
            print (n2,n1)
            fi = reg_inv2(k[:,:,n2,n1], lm)
            #ik_n1 = N.linalg.inv(k[n2,:,n2,:])
            pt = N.dot(fi, image[imslice]).trace()
            #pt = image[imslice]*reg_inv2(k[n2,n1], lm)
            #pt = image[imslice]*regularized_inverse(k[n2,n1], lm)
            #pt = image[imslice]*N.linalg.inv(k[n2,n1])
            #d[n2,n1] = pt.sum()
            d[n2,n1] = pt
    image[imslice] = d
