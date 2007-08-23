#!/usr/bin/env python
import gtk
import gobject
import os
import sys
import pylab as P
import numpy as N
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.backends.backend_gtkagg import \
     FigureCanvasGTKAgg as FigureCanvas
import matplotlib

from recon.imageio import readImage
from recon.slicerimage import SlicerImage, compose_xform
from recon.visualization import *
from recon import util
from odict import odict

ui_info = \
'''<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <menuitem action='Open Image'/>
      <separator/>
      <menuitem action='Quit'/>
    </menu>
    <menu action='ToolsMenu'>
      <menuitem action='Load Overlay'/>
      <menuitem action='Unload Overlay'/>
      <menuitem action='Overlay Adjustment Toolbox'/>
      <separator/>
      <menuitem action='Plot In Sliceview'/>
      <separator/>
      <menuitem action='Run Recon GUI'/>
    </menu>
  </menubar>
</ui>'''

cmap_lookup = odict((
    (0, P.cm.Blues),
    (1, P.cm.Greens),
    (2, P.cm.Greys),
    (3, P.cm.Oranges),
    (4, P.cm.Purples),
    (5, P.cm.Reds),
    (6, P.cm.Spectral),
    (7, P.cm.autumn),
    (8, P.cm.bone),
    (9, P.cm.cool),
    (10, P.cm.copper),
    (11, P.cm.gist_earth),
    (12, P.cm.gist_gray),
    (13, P.cm.gist_heat),
    (14, P.cm.gist_rainbow),
    (15, P.cm.gray),
    (16, P.cm.hot),
    (17, P.cm.hsv),
    (18, P.cm.jet),
    (19, P.cm.spring),
    (20, P.cm.summer),
    (21, P.cm.winter),
    ))

interp_types = ['nearest', 'bilinear', 'sinc']
interp_lookup = odict([(num,name) for num,name in enumerate(interp_types)])

##############################################################################
class spmclone (gtk.Window):
    
    def __init__(self, image, title="spmclone", parent=None):
        gtk.Window.__init__(self)
        children = self.get_children()
        if children:
            self.remove(children[0])            
            self.hide_all()
        table = gtk.Table(4, 2)

        # super dumb way of storing the image object with its original
        # class identification
        self.img_obj = image
        self.image = len(image.shape) > 3 \
                     and SlicerImage(image.subImage(0)) \
                     or SlicerImage(image)
        if iscomplex(self.image[:]):
            self.image.prefilter = abs_xform

        self.overlay_img = None
        self.zoom = 0
        self.slice_patches = None
        self.setNorm()
        self.dimlengths = self.image.dr * N.array(self.image.shape)
        zyx_lim = self.image.extents()
        # I'm using [ax,cor,sag] such that this list informs each
        # sliceplot what dimension it slices in the image array
        # (eg for coronal data, the coronal plot slices the 0th dim)        
        ax, cor, sag = self.image.slicing()
        origin = [0,0,0]
        # Make the ortho plots ---
        self.ax_plot=SlicePlot(self.image.data_xform(ax, origin), 0, 0, ax,
                               norm=self.norm,
                               extent=(zyx_lim[2] + zyx_lim[1]))
        
        self.cor_plot=SlicePlot(self.image.data_xform(cor, origin), 0, 0, cor,
                                norm=self.norm,
                                extent=(zyx_lim[2] + zyx_lim[0]))
        
        self.sag_plot=SlicePlot(self.image.data_xform(sag, origin), 0, 0, sag,
                                norm=self.norm,
                                extent=(zyx_lim[1] + zyx_lim[0]))
        
        # Although it doesn't matter 99% of the time, this list is
        # expected to be ordered this way
        self.sliceplots = [self.ax_plot, self.cor_plot, self.sag_plot]

        # menu bar
        merge = gtk.UIManager()
        merge.insert_action_group(self._create_action_group(), 0)
        mergeid = merge.add_ui_from_string(ui_info)
        self.menubar = merge.get_widget("/MenuBar")

        table.attach(self.menubar, 0, 2, 0, 1)
        self.menubar.set_size_request(600,30)
        table.attach(self.cor_plot, 0, 1, 1, 2)
        self.cor_plot.set_size_request(250,250)
        table.attach(self.sag_plot, 1, 2, 1, 2)
        self.sag_plot.set_size_request(250,250)
        table.attach(self.ax_plot, 0, 1, 2, 3)
        self.ax_plot.set_size_request(250,250)

        self.displaybox = DisplayInfo(self.image)
        self.displaybox.attach_toggle(self.crosshair_hider)
        self.displaybox.attach_imginterp(self.interp_handler)
        self.displaybox.attach_imgframe(self.zoom_handler)
        self.displaybox.attach_imgspace(self.rediculous_handler)
        table.attach(self.displaybox, 1, 2, 3, 4)
        self.displaybox.set_size_request(300,300)
        self.statusbox = DisplayStatus(tuple(self.image.vox_coords.tolist()),
                                       tuple(self.image.zyx_coords().tolist()),
                                       self.image[tuple(self.image.vox_coords)])
        table.attach(self.statusbox, 0, 1, 3, 4)
        self.statusbox.set_size_request(300,300)
        self.connect("configure_event", self.resize_handler)
        #table.set_row_spacing(1,25)
        # heights = 800
        # 250 plot 1
        # 250 plot 2
        # 370 info stuff
        # 30 menubar
        self.connect_crosshair_id = []
        self.connectCrosshairEvents()
        
        try:
            self.set_screen(parent.get_screen())
        except AttributeError:
            self.connect("destroy", lambda x: gtk.main_quit())
        self.set_data("ui-manager", merge)
        self.add_accel_group(merge.get_accel_group())
        self.set_default_size(600,730)
        self.set_border_width(3)
        self.set_title(title)
        self.add(table)        
        self.show_all()        
        #self.setUpAxesSize()        
        P.show()

    #-------------------------------------------------------------------------
    def setNorm(self):
        "sets the whitepoint and blackpoint (uses raw data, not scaled)"
        p01 = P.prctile(self.image[:], 1.0)
        p99 = P.prctile(self.image[:], 99.)
        self.norm = P.normalize(vmin = p01, vmax = p99)
        if hasattr(self, "overlay_img") and self.overlay_img:
            p01 = P.prctile(self.overlay_img[:], 1.0)
            p99 = P.prctile(self.overlay_img[:], 99.)
            self.overlay_norm = P.normalize(vmin = p01, vmax = p99)

    #-------------------------------------------------------------------------
    def xy(self, slice_idx):
        (ax, cor, sag) = self.image.slicing()
        x,y = {
            ax: (2, 1), # x,y
            cor: (2, 0), # x,z
            sag: (1, 0), # y,z
        }.get(slice_idx)
        return x,y
    #-------------------------------------------------------------------------
    def updateSlices(self, zyx, sliceplots=None, image=None, norm=None):
        if not sliceplots:
            sliceplots = self.sliceplots
        if not image:
            image = self.image
        if not norm:
            norm = self.norm
        for sliceplot in sliceplots:
            idx = sliceplot.slice_idx
            sliceplot.setData(image.data_xform(idx, zyx), norm=norm)
            if self.slice_patches is not None:
                p_idx = int(self.image.vox_coords[idx])
                sliceplot.showPatches(self.slice_patches[idx][p_idx])

    #-------------------------------------------------------------------------
    def updateCrosshairs(self):
        for s,sliceplot in enumerate(self.sliceplots):
            idx = sliceplot.slice_idx
            zyx = self.image.zyx_coords().tolist()
            zyx.pop(s)
            ud,lr = zyx
            sliceplot.setCrosshairs(lr,ud)

    #-------------------------------------------------------------------------
    def externalUpdate(self, new_img):
        old_vox = self.image.vox_coords
        if len(new_img.shape) > 3:
            self.image = SlicerImage(new_img.subImage(0))
        else:
            self.image = SlicerImage(new_img)
        self.image.vox_coords = old_vox
        if iscomplex(self.image[:]):
            self.image.prefilter = abs_xform
        self.setNorm()
        self.setUpAxesSize()
        self.updateSlices(self.image.zyx_coords())
        
    #-------------------------------------------------------------------------
    def setUpAxesSize(self):
        "Scale the axes appropriately for the image dimensions"
        # assume that image resolution is isotropic in dim2 and dim1
        # (not necessarily in dim0)
        # want the isotropic resolution plot to be 215x215 pixels
        #xy_imgsize = 215. # this should be more like 85% of the dim0 sliceplot
        ref_size = self.dimlengths[-1]
        slicing = self.image.slicing()
        idx = slicing.index(0)
        xy_imgsize = .85 * min(*self.sliceplots[idx].get_width_height())
        for sliceplot in self.sliceplots:
            dims_copy = self.dimlengths.tolist()
            ax = sliceplot.getAxes()
            s_idx = sliceplot.slice_idx
            dims_copy.remove(self.dimlengths[s_idx])
            slice_y, slice_x = self.image.is_xpose(s_idx) and \
                               dims_copy[::-1] or dims_copy
            height = xy_imgsize*slice_y/ref_size
            width = xy_imgsize*slice_x/ref_size
            canvas_x, canvas_y = sliceplot.get_width_height()
            w = width/canvas_x
            h = height/canvas_y
            l = (1.0 - width/canvas_x)/2.
            b = (1.0 - height/canvas_y)/2.
            ax.set_position([l,b,w,h])
            sliceplot.draw_idle()

    #-------------------------------------------------------------------------
    def connectCrosshairEvents(self, mode="enable"):
        if mode=="enable":
            self._dragging = False
            for sliceplot in self.sliceplots:
                self.connect_crosshair_id.append(sliceplot.mpl_connect(
                    "button_press_event", self.SPMouseDown))
                self.connect_crosshair_id.append(sliceplot.mpl_connect(
                    "button_release_event", self.SPMouseUp))
                self.connect_crosshair_id.append(sliceplot.mpl_connect(
                    "motion_notify_event", self.SPMouseMotion))
                sliceplot.toggleCrosshairs(mode=True)
        else:
            if len(self.connect_crosshair_id):
                for id,sliceplot in enumerate(self.sliceplots):
                    sliceplot.mpl_disconnect(self.connect_crosshair_id[id])
                    sliceplot.mpl_disconnect(self.connect_crosshair_id[id+1])
                    sliceplot.mpl_disconnect(self.connect_crosshair_id[id+2])
                    sliceplot.toggleCrosshairs(mode=False)
                self.connect_crosshair_id = []

    #-------------------------------------------------------------------------
    def SPMouseDown(self, event):
        # for a new mouse down event, reset the mouse positions
        self._mouse_lr = self._mouse_ud = None
        self._dragging = event.inaxes
        self.updateCoords(event)

    #-------------------------------------------------------------------------
    def SPMouseUp(self, event):
        # if not dragging, no business being here!
        if self._dragging:
            self.updateCoords(event)
            self._dragging = False

    #-------------------------------------------------------------------------
    def SPMouseMotion(self, event):
        if self._dragging:
            self.updateCoords(event)

    #-------------------------------------------------------------------------
    def updateCoords(self, event):
        "Update all the necessary sliceplot data based on a mouse click."
        # The tasks here are:
        # 1 find zyx_coords of mouse click and translate to vox_coords
        # 2 update the transverse sliceplots based on vox_coords
        # 2a update the transverse overlays if present
        # 3 update crosshairs on all sliceplots
        # 4 update voxel space and zyx space texts
        sliceplot = event.canvas
        # using terminology up-down, left-right to avoid confusion with y,x
        ud,lr = sliceplot.getEventCoords(event)
        if self._mouse_lr == lr and self._mouse_ud == ud:
            return
        if lr is None or ud is None:
            return
        self._mouse_lr, self._mouse_ud = (lr, ud)
        # trans_sliceplots are the transverse plots that get
        # updated from where the mouse clicked
        (ax, cor, sag) = self.image.slicing()
        trans_sliceplots = {
            self.sliceplots[ax]: (self.sliceplots[sag], self.sliceplots[cor]),
            self.sliceplots[cor]: (self.sliceplots[sag], self.sliceplots[ax]),
            self.sliceplots[sag]: (self.sliceplots[cor], self.sliceplots[ax]),
            }.get(sliceplot)
        trans_idx = (trans_sliceplots[0].slice_idx,
                     trans_sliceplots[1].slice_idx)
        
        # where do left-right and up-down cut across in zyx space?
        trans_ax = self.image.transverse_slicing(sliceplot.slice_idx)
        zyx_clicked = self.image.zyx_coords()
        zyx_clicked[trans_ax[0]] = lr
        zyx_clicked[trans_ax[1]] = ud
        vox = self.image.zyx2vox(zyx_clicked)

        self.image.vox_coords[trans_idx[0]] = vox[trans_idx[0]]
        self.image.vox_coords[trans_idx[1]] = vox[trans_idx[1]]
        self.updateSlices(zyx_clicked, sliceplots=trans_sliceplots)

        if self.overlay_img:
            # basically do the same thing over again wrt the overlay dims
            (ax_o, cor_o, sag_o) = self.overlay_img.slicing()
            trans_overlays = {
                self.sliceplots[ax]: (self.overlays[sag_o], self.overlays[cor_o]),
                self.sliceplots[cor]: (self.overlays[sag_o], self.overlays[ax_o]),
                self.sliceplots[sag]: (self.overlays[cor_o], self.overlays[ax_o]),
                }.get(sliceplot)
            trans_idx = (trans_overlays[0].slice_idx,
                         trans_overlays[1].slice_idx)            
            vox = self.overlay_img.zyx2vox(zyx_clicked)
            self.overlay_img.vox_coords[trans_idx[0]] = vox[trans_idx[0]]
            self.overlay_img.vox_coords[trans_idx[1]] = vox[trans_idx[1]]
            self.updateSlices(zyx_clicked, sliceplots=trans_overlays,
                              image=self.overlay_img,
                              norm=self.overlay_norm)

        self.updateCrosshairs()
        # make text to update the statusbox label's
        self.statusbox.set_vox_text(self.image.vox_coords[::-1].tolist())
        self.statusbox.set_zyx_text(self.image.zyx_coords()[::-1].tolist())
        point = tuple(self.image.vox_coords)
        self.statusbox.set_intensity_text(self.image[point])

    #-------------------------------------------------------------------------
    def rediculous_handler(self, cbox):
        #mode = cbox.get_active()==0 and "enable" or "disable"
        #self.connectCrosshairEvents(mode=mode)
        print "You've hit a useless button!"
        return

    #-------------------------------------------------------------------------
    def interp_handler(self, cbox):
        interp_method = interp_lookup[cbox.get_active()]
        for sliceplot in self.sliceplots:
            sliceplot.setInterpo(interp_method)

    #-------------------------------------------------------------------------
    def crosshair_hider(self, toggle):
        hidden = (not toggle.get_active())
        for sliceplot in self.sliceplots:
            sliceplot.toggleCrosshairs(mode=hidden)

    #-------------------------------------------------------------------------
    def zoom_handler(self, cbox):
        "Changes the view range of the sliceplots to be NxN mm"
        self.zoom = {
            0: 0,
            1: 160,
            2: 80,
            3: 40,
            4: 20,
            5: 10,
        }.get(cbox.get_active(), 0)
        r_center = self.image.zyx_coords()
        if self.zoom:
            r_neg = r_center - N.array([self.zoom/2.]*3)
            r_pos = r_center + N.array([self.zoom/2.]*3)
            zyx_lim = zip(r_neg, r_pos)
            self.dimlengths = N.array([self.zoom]*3)
            
        else:
            self.dimlengths = self.image.dr * N.array(self.image.shape)
            zyx_lim = self.image.extents()
            
        for plot in self.sliceplots:
            x,y = self.image.transverse_slicing(plot.slice_idx)
            plot.setXYlim(zyx_lim[x], zyx_lim[y])
            plot.setCrosshairs(r_center[x], r_center[y])
            
        self.updateSlices(self.image.zyx_coords())
        self.setUpAxesSize()
    #-------------------------------------------------------------------------
    def resize_handler(self, window, event):
        #print "got resize signal"
        self.setUpAxesSize()
    #-------------------------------------------------------------------------
    def initoverlay(self, action):
        image_filter = gtk.FileFilter()
        image_filter.add_pattern("*.hdr")
        image_filter.add_pattern("*.nii")
        image_filter.set_name("Recon Images")
        fname = ask_fname(self, "Choose file to overlay...", action="open",
                          filter=image_filter)
        if not fname:
            return
        try:
            img = readImage(fname, "nifti")
        except:
            img = readImage(fname, "analyze")
        if len(img.shape) > 3:
            self.overlay_img = SlicerImage(img.subImage(0))
        else:
            self.overlay_img = SlicerImage(img)
            
        img_dims = N.take(N.array(self.image.shape) * self.image.dr,
                          self.image.slicing())
        ovl_dims = N.take(N.array(self.overlay_img.shape) * self.overlay_img.dr,
                          self.overlay_img.slicing())
        if not (img_dims == ovl_dims).all():
            print img_dims, ovl_dims
            print "Overlay failed because physical dimensions do not align..."
            print "base image dimensions (zyx): [%3.1f %3.1f %3.1f] (mm)"%tuple(img_dims)
            print "overlay image dimenensions (zyx: [%3.1f %3.1f %3.1f] (mm)"%tuple(ovl_dims)
            return
        self.setNorm()
        (ax, cor, sag) = self.overlay_img.slicing()
        self.ax_overlay = OverLay(self.ax_plot, ax,
                                  norm=self.overlay_norm,
                                  interpolation=self.ax_plot.interpolation)
        self.cor_overlay = OverLay(self.cor_plot, cor,
                                   norm=self.overlay_norm,
                                   interpolation=self.cor_plot.interpolation)
        self.sag_overlay = OverLay(self.sag_plot, sag,
                                   norm=self.overlay_norm,
                                   interpolation=self.sag_plot.interpolation)
        self.overlays = [self.ax_overlay, self.cor_overlay, self.sag_overlay]
        self.updateSlices(self.image.zyx_coords(),
                          sliceplots=self.overlays,
                          image=self.overlay_img,
                          norm=self.overlay_norm)

    #-------------------------------------------------------------------------
    def launch_overlay_toolbox(self, action):
        if self.overlay_img is not None:
            if not hasattr(self, "overlay_tools") or not self.overlay_tools:
                self.overlay_tools = OverlayToolWin(self.overlays, self)
            else:
                self.overlay_tools.present()

    #-------------------------------------------------------------------------
    def killoverlay(self, action):
        if self.overlay_img is not None:
            for overlay in self.overlays:
                overlay.removeSelf()
            if hasattr(self, "overlay_tools") and self.overlay_tools:
                self.overlay_tools.destroy()
                del self.overlay_tools
            self.overlay_img = None
            self.overlay_norm = None

    #-------------------------------------------------------------------------
    def load_new_image(self, action):
        image_filter = gtk.FileFilter()
        image_filter.add_pattern("*.hdr")
        image_filter.add_pattern("*.nii")
        image_filter.set_name("Recon Images")
        fname = ask_fname(self, "Choose file to open...", action="open",
                          filter=image_filter)
        if not fname:
            return
        try:
            img = readImage(fname, "nifti")
        except:
            img = readImage(fname, "analyze")
        self.killoverlay(None)
        self.__init__(img)

    #-------------------------------------------------------------------------
    def launch_sliceview(self, action):
        from recon.visualization.sliceview import sliceview
        sliceview(self.image, parent=self)
    #-------------------------------------------------------------------------
    def launch_recon_gui(self, action):
        from recon.visualization.recon_gui import recon_gui
        recon_gui(image=self.img_obj, parent=self)

    #-------------------------------------------------------------------------
    def _create_action_group(self):
        entries = (
            ( "FileMenu", None, "_File" ),
            ( "Open Image", gtk.STOCK_OPEN, "_Open Image", "<control>O",
              "Opens and plots a new image", self.load_new_image ),
            ( "Quit", gtk.STOCK_QUIT,
              "_Quit", "<control>Q",
              "Quits",
              lambda action: self.destroy() ),
            ( "ToolsMenu", None, "_Tools" ),
            ( "Load Overlay", None, "_Load Overlay", "",
              "Load an image to overlay", self.initoverlay ),
            ( "Unload Overlay", None, "_Unload Overlay", "",
              "Unload the overlay", self.killoverlay ),
            ( "Overlay Adjustment Toolbox", None,
              "_Overlay Adjustment Toolbox", "",
              "launch overlay toolbox", self.launch_overlay_toolbox ),
            ( "Plot In Sliceview", None,
              "_Plot In Sliceview", "", "opens image in sliceview",
              self.launch_sliceview ),
            ( "Run Recon GUI", None, "_Run Recon GUI", "", "opens gui",
              self.launch_recon_gui ),
        )

        action_group = gtk.ActionGroup("WindowActions")
        action_group.add_actions(entries)
        return action_group


##############################################################################
class SlicePlot (FigureCanvas):
    "A Canvas class containing a 2D matplotlib plot"    
    #-------------------------------------------------------------------------
    def __init__(self, data, x, y, slice_idx, cmap=P.cm.gray,
                 norm=None, interpolation="bilinear", extent=None):
        self.norm = norm
        self.cmap = cmap        
        self.interpolation=interpolation
        self.slice_idx = slice_idx
        # extent should be static, so set it and leave it alone
        if not extent:
            y,x = data.shape[-2:]
            extent = [-x/2., x/2., -y/2., y/2.]
        self.extent = extent
        self.ylim = tuple(extent[2:])
        self.xlim = tuple(extent[:2])
        fig = P.Figure(figsize=P.figaspect(data), dpi=80)
        ax = fig.add_subplot(111)
        ax.yaxis.tick_right()
        ax.title.set_y(1.05) 
        FigureCanvas.__init__(self, fig)
        self.setData(data)
        self._init_crosshairs(x, y)


    #-------------------------------------------------------------------------
    def _init_crosshairs(self, x, y):
        self.x, self.y = x,y
        row_data, col_data = self._crosshairs_data(x, y)
        row_line = Line2D(row_data[0], row_data[1], color="r", alpha=.5)
        col_line = Line2D(col_data[0], col_data[1], color="r", alpha=.5)
        self.crosshairs = (row_line, col_line)
        ax = self.getAxes()
        ax.add_artist(row_line)
        ax.add_artist(col_line)

    #-------------------------------------------------------------------------
    def _crosshairs_data(self, x, y):
        ylim = self.getAxes().get_ylim()
        xlim = self.getAxes().get_xlim()
        data_width, data_height = (xlim[1]-xlim[0], ylim[1]-ylim[0])
        row_data = ((x+.5-data_width/4., x+.5+data_width/4.), (y+.5, y+.5))
        col_data = ((x+.5, x+.5), (y+.5-data_height/4., y+.5+data_height/4.))
        return row_data, col_data

    #------------------------------------------------------------------------- 
    def setCrosshairs(self, x, y):
        if x is not None: self.x = x
        if y is not None: self.y = y
        row_data, col_data = self._crosshairs_data(self.x, self.y)
        row_line, col_line = self.crosshairs
        row_line.set_data(*row_data)
        col_line.set_data(*col_data)
        self.draw_idle()

    #-------------------------------------------------------------------------
    def toggleCrosshairs(self, mode=True):
        for line in self.crosshairs:
            line.set_visible(mode)
        self.draw_idle()

    #-------------------------------------------------------------------------
    def getEventCoords(self, event):
        if event.xdata is not None: x = event.xdata
        else: x = None
        if event.ydata is not None: y = event.ydata
        else: y = None
        if x < self.extent[0] or x >= self.extent[1]:
            x = None
        if y < self.extent[2] or y >= self.extent[3]:
            y = None
        return (y,x)

    #-------------------------------------------------------------------------
    def getAxes(self):
        # let's say there's only 1 axes in the figure
        return self.figure.axes[0]

    #-------------------------------------------------------------------------
    def getImage(self, num=0):
        images = self.getAxes().images
        return len(images) > num and images[num] or None
        
    #-------------------------------------------------------------------------
    def setImage(self, image, num=0):
        if self.getImage(num=num):
            self.getAxes().images[num] = image
    #-------------------------------------------------------------------------
    def showPatches(self, patches):
        self.getAxes().patches = []
        for p in patches:
            self.getAxes().add_patch(p)
        self.draw_idle()
    #-------------------------------------------------------------------------
    def setCmap(self, cmapObj):
        self.setData(self.data, cmap=cmapObj)
    #-------------------------------------------------------------------------
    def setInterpo(self, interp_method):
        self.setData(self.data, interpolation=interp_method)
    #-------------------------------------------------------------------------
    def setXYlim(self, xlim, ylim):
        self.setData(self.data, ylim=ylim, xlim=xlim)
    
    #-------------------------------------------------------------------------
    def setData(self, data, norm=None, cmap=None,
                interpolation=None, ylim=None, xlim=None):
        ax = self.getAxes()
        if interpolation: self.interpolation = interpolation
        if cmap: self.cmap = cmap
        if norm: self.norm = norm
        if ylim: self.ylim = ylim
        if xlim: self.xlim = xlim
        if self.getImage() is None:
            ax.imshow(data, origin="lower", extent=self.extent)
        img = self.getImage()        
        img.set_data(data)
        img.set_cmap(self.cmap)
        img.set_interpolation(self.interpolation)
        img.set_norm(self.norm)
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        self.data = data
        self.draw_idle()

##############################################################################
class OverLay (object):
    """
    An Overlay inherits from the SlicePlot, and shares a Figure object
    with an existing SlicePlot.
    """

    def __init__(self, sliceplot, slice_idx, alpha=.45,
                 cmap=P.cm.gist_heat, norm=None, interpolation="nearest"):

        self.sliceplot = sliceplot
        self.norm = norm
        self.cmap = cmap
        self.alpha = alpha
        self.interpolation = interpolation
        self.slice_idx = slice_idx
        # set up the shared axes object for an overlay
        ax = sliceplot.getAxes()
        ax.hold(True)
        ax.set_axis_bgcolor('k')

    #-------------------------------------------------------------------------
    def setData(self, data, alpha=None, cmap=None,
                norm=None, interpolation=None):
        self.alpha = alpha or self.alpha
        self.cmap = cmap or self.cmap
        self.norm = norm or self.norm
        self.interpolation = interpolation or self.interpolation
        ax = self.sliceplot.getAxes()
        extent = self.sliceplot.getImage().get_extent()
        if self.sliceplot.getImage(num=1) is None:
            ax.imshow(data, extent=extent, origin="lower")
        # retrieve overlay image ref
        img = self.sliceplot.getImage(num=1)
        img.set_data(data)
        img.set_cmap(self.cmap)
        img.set_alpha(self.alpha)
        img.set_interpolation(self.interpolation)
        img.set_norm(self.norm)
        self.data = data
        self.sliceplot.draw_idle()

    #-------------------------------------------------------------------------
    def setAlpha(self, alphaVal):
        self.alpha = alphaVal
        self.setData(self.data)

    #-------------------------------------------------------------------------
    def setCmap(self, cmapObj):
        self.cmap = cmapObj
        self.setData(self.data)
    #-------------------------------------------------------------------------
    def setInterpo(self, interpo):
        if interpo in interp_lookup.values():
            self.interpolation = interpo
            self.setData(self.data)
    #-------------------------------------------------------------------------
    def removeSelf(self):
        ax = self.sliceplot.getAxes()
        if len(ax.images) > 1:
            ax.images.pop(1)
            self.sliceplot.draw_idle()
    
##############################################################################
class DisplayInfo (gtk.Frame):
    # height = 300
    # frame = 70
    # 5*labels = 30*5 = 150
    # large label = 80
    def __init__(self, image):
        vbox = gtk.VBox()
        
        # want 5 small labels, 1 larger label and a sub-frame
        dimlabel = gtk.Label("Dimensions: "+self.getdims(image))
        dtypelabel = gtk.Label("Datatype: "+image[:].dtype.name)
        scalelabel = gtk.Label("Intensity: %1.8f X"%image.scaling)
        voxlabel = gtk.Label("Vox size: "+self.getvox(image))
        originlabel = gtk.Label("Origin: "+self.getorigin(image))
        # make 4 lines worth of space for this label
        xform = image.orientation_xform.tomatrix()
        xformlabel = gtk.Label("Dir Cos: \n" + \
                               str(xform))
        xformlabel.set_size_request(300,80)


        buttons = gtk.Frame()
        buttons.set_size_request(300,70)
        buttons.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        frametable = gtk.Table(2,2)
        frametable.set_row_spacings(5)
        frametable.set_col_spacings(10)

        self.imgframe = gtk.combo_box_new_text()
        for size in ["Full Volume", "160x160x160mm", "80x80x80mm",
                     "40x40x40mm", "20x20x20mm", "10x10x10mm"]:
            self.imgframe.append_text(size)
        self.imgframe.set_active(0)
        
        self.imspace = gtk.combo_box_new_text()
        for space in ["World Space", "Voxel Space"]:
            self.imspace.append_text(space)
        self.imspace.set_active(0)
        
        self.imginterp = gtk.combo_box_new_text()
        for interp in interp_types:
            self.imginterp.append_text(interp)
        self.imginterp.set_active(1)

        self.hidecrosshairs = gtk.ToggleButton(label="Hide Crosshairs")
        
        frametable.attach(self.imgframe, 0, 1, 0, 1)
        frametable.attach(self.imspace, 0, 1, 1, 2)
        frametable.attach(self.imginterp, 1, 2, 0, 1)
        frametable.attach(self.hidecrosshairs, 1, 2, 1, 2)
        buttons.add(frametable)

        vbox.pack_start(dimlabel)
        vbox.pack_start(dtypelabel)
        vbox.pack_start(scalelabel)
        vbox.pack_start(voxlabel)
        vbox.pack_start(originlabel)
        vbox.pack_start(xformlabel)
        vbox.pack_start(buttons)
        gtk.Frame.__init__(self)
        self.set_border_width(5)        
        self.add(vbox)

    #-------------------------------------------------------------------------
    def getdims(self, image):
        return "%d x %d x %d"%image.shape[::-1]
    #-------------------------------------------------------------------------
    def getvox(self, image):
        return "%1.3f x %1.3f x %1.3f"%(image.xsize, image.ysize, image.zsize)
    #-------------------------------------------------------------------------
    def getorigin(self, image):
        return "%d x %d x %d"%(image.x0/image.xsize,
                               image.y0/image.ysize,
                               image.z0/image.zsize)
    #-------------------------------------------------------------------------
    def attach_imgframe(self, func):
        self.imgframe.connect("changed", func)
    #-------------------------------------------------------------------------
    def attach_imgspace(self, func):
        self.imspace.connect("changed", func)
    #-------------------------------------------------------------------------
    def attach_imginterp(self, func):
        self.imginterp.connect("changed", func)
    #-------------------------------------------------------------------------
    def attach_toggle(self, func):
        self.hidecrosshairs.connect("toggled", func)

    
##############################################################################
class DisplayStatus (gtk.Frame):

    def __init__(self, vox_coords, zyx_coords, intensity):
        main_table = gtk.Table(1,2)
        main_table.attach(gtk.Label(""), 0, 1, 1, 2)

        subframe = gtk.Frame()
        subframe.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        subframe.set_size_request(300,170)

        sub_table = gtk.Table(4,2)
        sub_table.attach(gtk.Label("mm:"), 0, 1, 0, 1)
        sub_table.attach(gtk.Label("vx:"), 0, 1, 1, 2)
        sub_table.attach(gtk.Label("intensity:"), 0, 1, 2, 3)
        sub_table.attach(gtk.Label("MNI:"), 0, 1, 3, 4)
        self.zyx_loc = gtk.Label("%2.1f %2.1f %2.1f"%zyx_coords)
        self.vx_loc = gtk.Label("%d %d %d"%vox_coords)
        self.intensity = gtk.Label("%3.4f"%intensity)
        self.mni_loc = gtk.Label("?? ?? ??")
        sub_table.attach(self.zyx_loc, 1, 2, 0, 1)
        sub_table.attach(self.vx_loc, 1, 2, 1, 2)
        sub_table.attach(self.intensity, 1, 2, 2, 3)
        sub_table.attach(self.mni_loc, 1, 2, 3, 4)
        subframe.add(sub_table)

        main_table.attach(subframe, 0, 1, 0, 1)
        gtk.Frame.__init__(self)
        self.add(main_table)

    #-------------------------------------------------------------------------
    def set_zyx_text(self, locs):
        text = "%2.1f %2.1f %2.1f"%tuple(locs)
        self.zyx_loc.set_text(text)
    #-------------------------------------------------------------------------
    def set_vox_text(self, locs):
        text = "%d %d %d"%tuple(locs)
        self.vx_loc.set_text(text)
    #-------------------------------------------------------------------------
    def set_intensity_text(self, val):
        text = "%3.4f"%val
        self.intensity.set_text(text)

##############################################################################
class OverlayToolWin (gtk.Window):
    "A Window class defining a pop-up control widget for the overlay plot."

    def __init__(self, overlay_list, parent):
        self.padre = parent
        self.overlay_ref = overlay_list
        self.vbox = gtk.VBox(spacing=5)
        self.vbox.set_border_width(10)        
        alpha = self.overlay_ref[0].alpha
        interp = self.overlay_ref[0].interpolation
        # add alpha slider and label
        self.alphaslider = gtk.HScale(gtk.Adjustment(alpha, 0.05, 1, .05, .05))
        self.alphaslider.set_digits(3)
        self.alphaslider.set_value_pos(gtk.POS_RIGHT)
        self.alphaslider.get_adjustment().connect("value-changed",
                                                  self.alpha_handler)
        self.vbox.pack_start(gtk.Label("Opaqueness level"), False, False, 0)
        self.vbox.pack_start(self.alphaslider)

        self.vbox.pack_start(gtk.HSeparator(), expand=False)
        
        # add cmap combo box and label
        self.cmap_list = gtk.combo_box_new_text()
        cmaps = [cmap.name for cmap in cmap_lookup.values()]
        for cmap in cmaps:
            self.cmap_list.append_text(cmap)
        self.cmap_list.set_active(cmaps.index("gist_heat"))
        self.cmap_list.connect("changed", self.cmap_handler)
        self.vbox.pack_start(gtk.Label("Overlay colormap"),
                             False, False, 0)
        self.vbox.pack_start(self.cmap_list)

        # add interpolation combo box and label
        self.interpo_list = gtk.combo_box_new_text()
        for interpo in interp_lookup.values():
            self.interpo_list.append_text(interpo)
        self.interpo_list.set_active(interp_lookup.values().index(interp))
        self.interpo_list.connect("changed", self.interpo_handler)
        self.vbox.pack_start(gtk.Label("Overlay interpolation"),
                             False, False, 0)
        self.vbox.pack_start(self.interpo_list)
        gtk.Window.__init__(self)
        self.set_destroy_with_parent(True)
        self.connect("destroy", self._takedown)
        #self.set_default_size(250,150)
        self.set_title("Overlay Plot Controls")
        self.add(self.vbox)
        self.show_all()
        P.show()

    #-------------------------------------------------------------------------
    def _takedown(self, foo):
        self.padre.overlay_tools = None
        foo.destroy()

    #-------------------------------------------------------------------------
    def alpha_handler(self, adj):
        for overlay in self.overlay_ref:
            overlay.setAlpha(self.alphaslider.get_value())

    #-------------------------------------------------------------------------
    def cmap_handler(self, cbox):
        cmap = cmap_lookup[cbox.get_active()]
        for overlay in self.overlay_ref:
            overlay.setCmap(cmap)

    #-------------------------------------------------------------------------
    def interpo_handler(self, cbox):
        interpo = interp_lookup[cbox.get_active()]
        for overlay in self.overlay_ref:
            overlay.setInterpo(interpo)


##############################################################################
if __name__ == "__main__":
    fname = sys.argv[1]
    ftype = len(sys.argv) > 2 and sys.argv[2] or "analyze"
    img = readImage(fname, ftype)
    spmclone(img)
    
