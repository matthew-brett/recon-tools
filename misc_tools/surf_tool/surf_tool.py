#!/usr/bin/env python

import gtk

from recon import imageio, util
from recon.operations.ReorderSlices import ReorderSlices

from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
import pylab as P
import matplotlib.axes3d as P3
import matplotlib as M
import numpy as N

class surfplot (gtk.Window):

    def __init__(self, data, dimarrays=None,
                 dimnames=("phase","slice","read-out"),
                 zextent=None, title=None):
        self.data = data
        n_slice = data.shape[-3]

        if dimarrays is None:
            self.Xx, self.Yy = P.meshgrid(N.arange(self.data.shape[-1]),
                                          N.arange(self.data.shape[-2]))
        else:
            self.Xx, self.Yy = dimarrays

        self.dimnames = dimnames
        hbox = gtk.HBox()

        self.fig = M.figure.Figure()
        self.surf_plot = FigureCanvas(self.fig)
        self.surf_plot.set_size_request(600,400)
        self.ax3d = P3.Axes3D(self.fig)

        if zextent is None:
            imin, imax = data[:].min(), data[:].max()
            eps = (imax - imin)*0.15
            self.zmin, self.zmax = imin-eps, imax+eps
        else:
            self.zmin, self.zmax = zextent
        print self.zmin, self.zmax
        #self.ax3d.set_zlim(imin-eps, imax+eps)
        hbox.pack_start(self.surf_plot)

        self.slice_slider = gtk.HScale(gtk.Adjustment(0, 0, n_slice,
                                                      1, 1, 1))
        self.slice_slider.set_digits(2)
        self.slice_slider.set_value_pos(gtk.POS_RIGHT)
        self.slice_slider.get_adjustment().connect("value-changed",
                                                   self.slice_handler)
        self.slice_slider.set_size_request(400,60)
        hbox.pack_start(self.slice_slider)
        gtk.Window.__init__(self)
        self.connect("destroy", lambda x: gtk.main_quit())
        self.set_default_size(600,460)
        self.add(hbox)
        if title is not None:
            self.set_title(title)
        self.show_all()
        self.set_surface((self.Xx,self.Yy,self.data[0]), self.dimnames)      
        P.show()

    def slice_handler(self, adj):
        dimarrays = (self.Xx, self.Yy, self.data[adj.get_value()])
        self.set_surface(dimarrays, self.dimnames)

    def set_surface(self, dimarrays, dimnames):
##         Rx,Px = P.meshgrid(N.arange(self.data.shape[-1]),
##                            N.arange(self.data.shape[-2]))
        if len(self.ax3d.collections):
            self.ax3d.collections.pop(0)
        
        self.ax3d.plot_wireframe(*dimarrays)
        self.ax3d.set_zlim(self.zmin, self.zmax)
        #self.ax3d.set_zlim(-8,5)
        self.ax3d.set_zlabel(dimnames[0])
        self.ax3d.set_ylabel(dimnames[1])
        self.ax3d.set_xlabel(dimnames[2])
        self.surf_plot.draw()

class FID_ref_pdiff_plot(surfplot):
    def __init__(self, fidname, zextent=None, vrange=None, rtype="bal"):
        brs1 = imageio.readImage(fidname, "fid", vrange=vrange)
        ReorderSlices().run(brs1)
        rvol = brs1.ref_data
        if rtype == "bal":
            inv_ref = util.ifft(rvol[0]) * N.conjugate(util.ifft(util.reverse(rvol[1], axis=-1)))
        elif rtype == "unbal":
            conj_order = N.arange(brs1.shape[-2])
            xleave = brs1.nseg
            util.shift(conj_order, -xleave)
            inv_ref = util.ifft(rvol[0]) * \
                      N.conjugate(N.take(util.ifft(rvol[0]), conj_order, axis=-2))
        else:
            inv_ref = util.ifft(rvol[0])
        phs_vol = util.unwrap_ref_volume(inv_ref, 0, 64)
            
        S = brs1.shape[-3]
        acq_order = brs1.acq_order
        s_ind = N.concatenate([N.nonzero(acq_order==s)[0] for s in range(S)])
        pss = N.take(brs1.slice_positions, s_ind)
        ro_line = N.arange(brs1.shape[-1]) - (brs1.shape[-1]/2)
        Ro,Sp = P.meshgrid(ro_line,pss)
        surfplot.__init__(self, N.swapaxes(phs_vol, 0, 1), dimarrays=(Ro, Sp),
                          zextent=zextent, title=fidname)


class fmap_surf_plot(surfplot):
    def __init__(self, fmapname):
        fmap = imageio.readImage(fmapname, "nifti")
        surfplot.__init__(self, fmap.subImage(0), title=fmapname)

class Siemens_ref_pdiff_plot(surfplot):
    "takes a filename indicating the ref phase data block file"
    def __init__(self, dblock_name, dblock_shape, dtype=N.complex128,
                 chan=0, vol=0, xstart=None, xstop=None):
        dblock = N.fromstring(open(dblock_name).read(), dtype=dtype)
        print dblock.shape
        print dblock_shape
        (S,U,R) = dblock_shape[-3:]
        dblock.shape = dblock_shape
        dblock = dblock[chan,vol]
        phs_vol = get_unbal_pdiffs(dblock, xstart, xstop)
        xstart = xstart or 0
        xstop = xstop or R
        r_vec = N.arange(xstart,xstop) - R/2
        Ro,S = P.meshgrid(r_vec, N.arange(S))
        surfplot.__init__(self, N.swapaxes(phs_vol, 0, 1), dimarrays=(Ro,S),
                          title=dblock_name)
        

def get_unbal_pdiffs(data, xstart=None, xstop=None):
    S,U,R = data.shape
    rslice = (slice(None), slice(0,U-1), slice(None))
    cslice = (slice(None), slice(1,U), slice(None))
    iref = util.ifft(data[rslice]) * N.conjugate(util.ifft(data[cslice]))
    return util.unwrap_ref_volume(iref, xstart, xstop)
    

if __name__ == "__main__":
    import sys
    print sys.argv
    if len(sys.argv) < 3:
        print "basic usage: surf_tool.py surface-type file-name"
    elif sys.argv[1] == "fid-pdiff":
        fidname = sys.argv[2]
        zx = None
        vrange = None
        rtype = "bal"
        if len(sys.argv) > 3:
            rtype = sys.argv[3]
        if len(sys.argv) > 4:
            extents = sys.argv[4].split(":")
        zx = tuple(map(int, extents))
        if len(sys.argv) > 5:
            vrange = sys.argv[5].split(":")
            vrange = tuple(map(int, vrange))
        FID_ref_pdiff_plot(fidname, zextent=zx, vrange=vrange, rtype=rtype)
    elif sys.argv[1] == "fmap":
        fmap_surf_plot(sys.argv[2])
    elif sys.argv[1] == "siemens-pdiff":
        block_name = sys.argv[2]
        chan, vol = 0, 0
        dtype = N.complex128
        ro_range = (None, None)
        block_shape = tuple(map(int, (sys.argv[3], sys.argv[4], sys.argv[5],
                                      sys.argv[6], sys.argv[7])))

        if len(sys.argv) > 9:
            chan, vol = map(int, (sys.argv[8], sys.argv[9]))
        
        if len(sys.argv) > 10:
            ro_range = sys.argv[10].split(':')
            ro_range = map(int, ro_range)

        if len(sys.argv) > 11:
            dtype = { "z": N.complex128,
                      "c": N.complex64,
                      }.get(sys.argv[11])

        Siemens_ref_pdiff_plot(block_name, block_shape, dtype=dtype,
                               chan=chan, vol=vol,
                               xstart=ro_range[0], xstop=ro_range[1])
    
    else:
        print "did not understand plot type"
        
    
