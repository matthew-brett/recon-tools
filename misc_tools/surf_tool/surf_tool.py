#!/usr/bin/env python

import gtk

from recon.imageio import readImage


from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
import pylab as P
import matplotlib.axes3d as P3
import matplotlib as M
import numpy as N

class fmapsurf (gtk.Window):

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
        

if __name__ == "__main__":
    import sys
    print sys.argv
    fmap = readImage(sys.argv[1], "nifti")
    fmapsurf(fmap.subImage(0))
