import numpy as N
from scipy.special.basic import lpmn

def _sph_harmonic_cart(m,n,l,x,y,z,r0):
    """inputs of (m,n,theta,phi) returns spherical harmonic of order
    m,n (|m|<=n) and argument theta and phi:  Y^m_n(theta,phi)

    n = order
    m = degree
    l = cos/sin toggle
    r0 = reference radius
    phi = latitudinal [0,pi]
    theta = longitudinal [0,2pi)
    """
    theta,phi,r = cart2polar(x,y,z)
    w = z/r #should = N.cos(phi)
    m,n = int(m), int(n)
    Pmn,Pmnd = lpmn(m,n,w)
    val = Pmn[-1,-1]
##     val *= sqrt((2*n+1)/4.0/pi)
##     val *= exp(0.5*(gammaln(n-m+1)-gammaln(n+m+1)))
##     val *= exp(1j*m*theta)
    val *= N.power(r/r0,n)
    if l == 1: val *= N.cos(m*theta)
    if l == 2: val *= N.sin(m*theta)
    return val

#sph_harm_cart = N.vectorize(_sph_harmonic_cart,'D')


def cart2polar(x,y,z):
    r = N.sqrt(x**2 + y**2 + z**2)
    phi = N.arccos(z/r)
    theta = N.arctan2(y,x)
    return theta,phi,r

def xyz_harms(xdim=64,ydim=64,zdim=20,dx=3.5,dy=3.5,dz=4.0,tab_num=300):
    x_table_300 = {
        '111': 1,
        '311': 0 - .01312,
        '511': 0 - .00747,
        '711': 0 + .00089,
        '911': 0 - .00006,
    }

    x_table_400 = {
        '111': 1,
        '311': 0 - .02332,
        '511': 0 - .02363,
        '711': 0 + .00500,
        '911': 0 - .00056,
    }

    x_table_500 = {
        '111': 1,
        '311': 0 - .03786,
        '511': 0 - .05585,
        '711': 0 + .01838,
        '911': 0 - .00327,
    }

    y_table_300 = {
        '112': 1,
        '312': 0 - .01363,
        '512': 0 - .00724,
        '712': 0 + .00086,
        '912': 0 - .00005,
    }

    y_table_400 = {
        '112': 1,
        '312': 0 - .02423,
        '512': 0 - .02288,
        '712': 0 + .00482,
        '912': 0 - .00054,
    }

    y_table_500 = {
        '112': 1.,
        '312': 0 - .03786,
        '512': 0 - .05585,
        '712': 0 + .01838,
        '912': 0 - .00327,
    }

    z_table_300 = {
        '100': 1,
        '300': 0 - .00304,
        '500': 0 - .02068,
        '700': 0 + .00322,
        '900': 0 - .00031,
    }

    z_table_400 = {
        '100': 1,
        '300': 0 - .00541,
        '500': 0 - .06537,
        '700': 0 + .01809,
        '900': 0 - .00306,
    }

    z_table_500 = {
        '100': 1.,
        '300': 0 - .00845,
        '500': 0 - .15960,
        '700': 0 + .06899,
        '900': 0 - .01830,
    }
    
    # say FOV is 224x224x80 in x, y, z
    # x,y in [-112,112), z in [-40,40)
    x = dx*(N.arange(xdim)-xdim/2)/100.
    y = dy*(N.arange(ydim)-ydim/2)/100.
    z = dz*(N.arange(zdim)-zdim/2)/100
    print x
    print y
    print z
    hx = N.zeros((zdim,ydim,xdim))
    hy = N.zeros((zdim,ydim,xdim))
    hz = N.zeros((zdim,ydim,xdim))
    (x_table, y_table, z_table, r0) = \
              {300: (x_table_300, y_table_300, z_table_300, 1.5),
               400: (x_table_400, y_table_400, z_table_400, 2.0),
               500: (x_table_500, y_table_500, z_table_500, 2.5),
               }.get(tab_num, (None,)*4)
    if not r0:
        print "bad tab_num: %d"%tab_num
        return (None,)*3

    for s in range(zdim):
        for r in range(ydim):
            for c in range(xdim):
                for key in x_table.keys():
                    n,m,l = map(int, key)
                    if n==1:
                        continue
                    hx[s,r,c] += \
                        _sph_harmonic_cart(m,n,l,x[c],y[r],z[s],r0)*x_table[key]

                for key in y_table.keys():
                    n,m,l = map(int, key)
                    if n==1:
                        continue
                    hy[s,r,c] += \
                        _sph_harmonic_cart(m,n,l,x[c],y[r],z[s],r0)*y_table[key]
                for key in z_table.keys():
                    n,m,l = map(int, key)
                    if n==1:
                        continue
                    hz[s,r,c] += \
                        _sph_harmonic_cart(m,n,l,x[c],y[r],z[s],r0)*z_table[key]
    return hx,hy,hz



