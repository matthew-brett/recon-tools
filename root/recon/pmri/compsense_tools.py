import numpy as np, pylab as P
from recon.pmri import grappa_sim

# again.. image is taken to be ([nchan], N2, N1)

def plot_power_law(X):
    xabs = np.abs(X).flat[:]
    xabs.sort()
    P.semilogy(xabs[::-1])
    P.show()

def l1_norm(X):
    if len(X.shape)==2:
        return np.abs(X).sum()
    else:
        return np.abs(X).sum(axis=-1).sum(axis=-1)

def tv_norm(X, abs=True):
    nr, nc = X.shape[-2:]
    f = lambda x: x
    if abs and X.dtype in ('D', 'F'):
        f = lambda x, g=f: np.abs(g(x))
    dy = np.zeros(X.shape, 'd')
    dy_sl = (slice(None),)*(len(X.shape)-2) + (slice(0,nr-1), slice(None))
    dx_sl = (slice(None),)*(len(X.shape)-2) + (slice(None), slice(0,nc-1))
    dx = np.zeros(X.shape, 'd')
    dy[dy_sl] = np.diff(f(X), axis=-2)
    dx[dx_sl] = np.diff(f(X), axis=-1)
    if dx.dtype.char in ('D', 'F'):
        tv = dx.real**2
        np.add(tv, dx.imag**2, tv)
        np.add(tv, dy.real**2, tv)
        np.add(tv, dy.imag**2, tv)
    else:
        tv = dx**2
        np.add(tv, dy**2, tv)

    np.sqrt(tv, tv)
    if len(X.shape)==2:
        return tv.sum()
    else:
        return tv.sum(axis=-1).sum(axis=-1)
    
def fourier_operator(N,M):
    """Defines the forward fourier operator, where rows are
    D[n,:] exp(-1j*2PI*n*m/N) for n,m in {-N/2, ..., N/2-1}
    """
    D = np.zeros((N,M), 'D')
    D = np.exp(-2j*np.pi*(np.arange(-N/2,N/2)[:,None] * np.arange(-M/2,M/2))/N)
    np.divide(D, np.sqrt(N), D)
    return D

def fourier_operator_blocks(N,M):
    """Defines a block diagonal operator such that each block transforms
    M points at a time in an N*M vector.
    """
    D = fourier_operator(N,M)
    D1D = np.zeros((N*M,N*M), 'D')
    for m in range(M):
        bslice = (slice(m*N, (m+1)*N), slice(m*M, (m+1)*M))
        D1D[bslice] = D
    return D1D

def pmri_sampling_operator(N2, N1, a=2):
    """Defines a block diagonal operators shaped (N2*N1/a, N2*N1) that
    subsamples a full (N2*N1) vector.
    """
    nsamp = (N1*N2)/a
    idt = np.eye(N1)
    M = np.zeros((nsamp, N2*N1), 'd')
    samps = grappa_sim.smash_lines_decomp(N2, a, 1)[0]
    for i, n2 in enumerate(samps):
        bslice = (slice(i*N1, (i+1)*N1), slice(n2*N1, (n2+1)*N1))
        M[bslice] = idt
    return M

def assert_rows_orthogonal(M):
    nr = M.shape[0]
    for r in range(nr-1):
        for rp in range(r+1, nr):
            assert np.abs(np.dot(M[r], M[rp].conjugate())) < 1e-10, "rows %d, %d not orthogonal"%(r, rp)


