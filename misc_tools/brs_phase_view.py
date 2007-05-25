#!/usr/bin/env python
import numpy as N, pylab as P
from recon import util, imageio
from recon.operations.ReorderSlices import ReorderSlices
from fmap_tool import fmapsurf

def viewphase(fidname, zextent=None, vrange=None, rtype="bal"):
    brs1 = imageio.readImage(fidname, "fid", vrange=vrange)
    ReorderSlices().run(brs1)
    rvol = brs1.ref_data
    if rtype == "bal":
        inv_ref = util.ifft(rvol[0]) * N.conjugate(util.ifft(util.reverse(rvol[1], axis=-1)))
    elif rtype == "unbal":
        conj_order = N.arange(brs1.shape[-2])
        xleave = brs1.nseg
        util.shift(conj_order, 0, -xleave)
        inv_ref = util.ifft(rvol[0]) * \
                  N.conjugate(N.take(util.ifft(rvol[0]), conj_order, axis=-2))
    else:
        inv_ref = util.ifft(rvol[0])
    phs_vol = util.unwrap_ref_volume(inv_ref, 0, 64)
    #phs_vol = N.angle(inv_ref)
    S = brs1.shape[-3]
    acq_order = brs1.acq_order
    s_ind = N.concatenate([N.nonzero(acq_order==s)[0] for s in range(S)])
    pss = N.take(brs1.slice_positions, s_ind)
    ro_line = N.arange(brs1.shape[-1]) - (brs1.shape[-1]/2)
    Ro,Sp = P.meshgrid(ro_line[:52],pss)
    fmapsurf(N.swapaxes(phs_vol[...,:52], 0, 1),
             dimarrays=(Ro, Sp), zextent=zextent, title=fidname)
    


if __name__ == "__main__":
    import sys
    print sys.argv
    zx = None
    vrange = None
    rtype = "bal"
    if len(sys.argv) > 2:
        rtype = sys.argv[2]
    if len(sys.argv) > 3:
        extents = sys.argv[3].split(":")
        zx = tuple(map(int, extents))
    if len(sys.argv) > 4:
        vrange = sys.argv[4].split(":")
        vrange = tuple(map(int, vrange))
    viewphase(sys.argv[1], zextent=zx, vrange=vrange, rtype=rtype)
