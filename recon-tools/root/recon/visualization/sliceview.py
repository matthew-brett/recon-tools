"The sliceview module defines classes providing a slice-plotting GUI"

import gtk
import gobject
import os
import pylab as P
import numpy as N
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_gtkagg import \
  FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import \
     NavigationToolbar2GTKAgg as NavigationToolbar
import matplotlib

from recon.imageio import readImage, ReconImage
from recon.slicerimage import SlicerImage, compose_xform
from recon.visualization import *
from odict import odict

ui_info = \
'''<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <menuitem action='Save Image'/>
      <menuitem action='Save Montage'/>
      <menu action='MoviesMenu'>
        <menuitem action='Slice movie'/>
        <menuitem action='Time movie'/>
      </menu>
      <separator/>
      <menuitem action='Quit'/>
    </menu>
    <menu action='ToolsMenu'>
      <menu action='SizeMenu'>
        <menuitem action='1x'/>
        <menuitem action='2x'/>
        <menuitem action='4x'/>
        <menuitem action='6x'/>        
        <menuitem action='8x'/>
      </menu>
      <menu action='ColorMapping'>
        <menuitem action='Blues'/>
        <menuitem action='Greens'/>
        <menuitem action='Greys'/>
        <menuitem action='Oranges'/>
        <menuitem action='Purples'/>
        <menuitem action='Reds'/>
        <menuitem action='Spectral'/>
        <menuitem action='autumn'/>
        <menuitem action='bone'/>
        <menuitem action='cool'/>
        <menuitem action='copper'/>
        <menuitem action='gist_earth'/>
        <menuitem action='gist_gray'/>
        <menuitem action='gist_heat'/>
        <menuitem action='gist_rainbow'/>
        <menuitem action='gray'/>
        <menuitem action='hot'/>
        <menuitem action='hsv'/>
        <menuitem action='jet'/>
        <menuitem action='spring'/>
        <menuitem action='summer'/>
        <menuitem action='winter'/>
      </menu>
      <menu action='Interpolation'>
        <menuitem action='nearest'/>
        <menuitem action='bilinear'/>
        <menuitem action='bicubic'/>
        <menuitem action='spline16'/>
        <menuitem action='spline36'/>
        <menuitem action='hanning'/>
        <menuitem action='hamming'/>
        <menuitem action='hermite'/>
        <menuitem action='kaiser'/>
        <menuitem action='quadric'/>
        <menuitem action='catrom'/>
        <menuitem action='gaussian'/>
        <menuitem action='bessel'/>
        <menuitem action='mitchell'/>
        <menuitem action='sinc'/>
        <menuitem action='lanczos'/>
        <menuitem action='blackman'/>
      </menu>
      <menuitem action='Contour Plot'/>
      <separator/>
      <menuitem action='View Ortho Plots'/>
      <separator/>
      <menuitem action='Run Recon GUI'/>
    </menu>
    <menu action='OverlayMenu'>
      <menuitem action='Load Overlay'/>
      <menuitem action='Unload Overlay'/>
      <menuitem action='Overlay Adjustment Toolbox'/>
    </menu>
    <menu action='Orientation'>
      <menuitem action='Native'/>
      <menuitem action='Axial'/>
      <menuitem action='Coronal'/>
      <menuitem action='Saggital'/>
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

interpo_methods = ['nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
                      'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                      'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
                      'lanczos', 'blackman']

interpo_lookup = odict([(num, name) for num,name in enumerate(interpo_methods)])

##############################################################################
class sliceview (gtk.Window):
    "A Window class containing various plots and widgets"
    
    #mag_norm = normalize()
    #phs_norm = normalize(-pi, pi)
    _mouse_x = _mouse_y = None
    _dragging = False

    #-------------------------------------------------------------------------
    def __init__(self, data, dim_names=[], title="sliceview",
                 cmap=P.cm.bone, parent=None):
        if isinstance(data, ReconImage):
            self.data = SlicerImage(data)
            # super dumb way of storing the image object with its original
            # class identification
            self.img_obj = data
        else:
            self.data = SlicerImage(N.asarray(data))
            self.img_obj = None
        self.overlay_data = None
        self.orient_mode = -1
        # if data is complex, show the magnitude by default
        self.iscomplex = iscomplex(self.data[:])
        self.transform = self.iscomplex and abs_xform or ident_xform
        # widget layout table
        table = gtk.Table(4, 2)        

        # control panel
        self.control_panel = \
          ControlPanel(self.data.shape, dim_names, self.iscomplex)
        self.control_panel.connect(
            self.radioHandler,
            self.sliderHandler,
            self.contrastHandler)
        self.control_panel.set_size_request(200, 200)
        table.attach(self.control_panel, 0, 1, 1, 2, xoptions=0, yoptions=0)

        # row plot
        self.rowplot = RowPlot(self.getRow())
        self.rowplot.set_size_request(400, 200)
        table.attach(self.rowplot, 1, 2, 1, 2, xoptions=0, yoptions=0)

        # column plot
        self.colplot = ColPlot(self.getCol())
        self.colplot.set_size_request(200, 400)
        table.attach(self.colplot, 0, 1, 2, 3, xoptions=0, yoptions=0)
        
        # Set up normalization BEFORE plotting images.
        # Contrast level of 1.0 gives default normalization (changed by
        # contrast slider).
        self.norm = None
        self.overlay_norm = None
        self.contrast = 1.0
        self.setNorm()
        self.updateDataRange()
        
        # slice image
        self.scrollwin = gtk.ScrolledWindow()
        self.scrollwin.set_border_width(0)
        self.scrollwin.set_policy(hscrollbar_policy=gtk.POLICY_AUTOMATIC,
                             vscrollbar_policy=gtk.POLICY_AUTOMATIC)
        self.scrollwin.set_size_request(400,400)
        self.sliceplot = SlicePlot(self.getSlice(),
          self.control_panel.getRowIndex(),
          self.control_panel.getColIndex(),
          cmap=cmap, norm=self.norm)

        self.connectFigureCanvasEvents(mode="init")
        self.auto_scale_image()
        self.scrollwin.add_with_viewport(self.sliceplot)
        #table.attach(self.sliceplot, 1, 2, 2, 3)
        table.attach(self.scrollwin, 1, 2, 2, 3)

        # status
        self.status = StatusBar(self.sliceplot, self.sliceDataRange(),
                                cmap=cmap, norm=self.norm)
        self.status.connect_button(self.ROIHandler)
        self.status.set_size_request(600,85)
        table.attach(self.status, 0, 2, 3, 4, xoptions=0, yoptions=0)

        # tool-bar
        merge = gtk.UIManager()
        merge.insert_action_group(self._create_action_group(self.image_scale),
                                  0)

        try:
            mergeid = merge.add_ui_from_string(ui_info)
        except gobject.GError, msg:
            print "building menus failed: %s" % msg
        self.menubar = merge.get_widget("/MenuBar")
        table.attach(self.menubar, 0, 2, 0, 1, yoptions=0)

        # initialize contour tools
        self.contour_tools = None
        
        # main window
        gtk.Window.__init__(self)
        try:
            self.set_screen(parent.get_screen())
            self.destroy_handle = self.connect('destroy',
                                               lambda x: parent.plotter_died())
        except AttributeError:
            self.connect("destroy", lambda x: gtk.main_quit())
        self.set_data("ui-manager", merge)
        self.add_accel_group(merge.get_accel_group())        
        # sum of widget height:
        # 27 for menu bar
        # 200 for row-plot, control panel
        # 400 for col-plot, scroll window
        # 40 for status bar (85)
        # = 670
        self.set_default_size(600,715)
        self.set_title(title)
        self.set_border_width(3)
        self.add(table)
        self.show_all()
        P.show()
        #gtk.main()

    #-------------------------------------------------------------------------
    def getRow(self):
        return self.getSlice()[self.control_panel.getRowIndex(),:]

    #-------------------------------------------------------------------------
    def getCol(self):
        return self.getSlice()[:,self.control_panel.getColIndex()]

    #-------------------------------------------------------------------------
    def getSlice(self):
        return self.transform(self.data[self.control_panel.getIndexSlices()])

    #-------------------------------------------------------------------------
    def getOverlaySlice(self):
        if self.overlay_data is not None:
            slices = self.control_panel.getIndexSlices()[-3:]
            return self.transform(self.overlay_data[slices])
        
    #-------------------------------------------------------------------------
    def updateRow(self):
        self.updateCrosshairs()
        self.rowplot.setData(self.getRow())

    #-------------------------------------------------------------------------
    def updateCol(self):
        self.updateCrosshairs()
        self.colplot.setData(self.getCol())
    #-------------------------------------------------------------------------
    def popup(self, event):
        win = gtk.Window()
        win.connect("destroy", lambda x: x.destroy())
        win.set_default_size(400,300)
        vbox = gtk.VBox()
        win.add(vbox)
        fig = P.Figure()
        ax = fig.add_subplot(111)
        ax.hold(True)
        if event.canvas == self.rowplot:
            rdata = self.getRow()
        else:
            rdata = self.getCol()
        ax.plot(rdata, label="data")
        if self.overlay_data is not None:
            if event.canvas == self.rowplot:
                rdata2 = self.getOverlaySlice()[self.control_panel.getRowIndex(),:]
            else:
                rdata2 = self.getOverlaySlice()[:,self.control_panel.getColIndex()]
            ax.plot(rdata2, label="overlay data")
        #ax.plot(rdata2/rdata, label="ovr/data")
        ax.legend(loc='upper left')
        ax.set_xlim(0, rdata.shape[0])
        #ax.set_ylim(*self.figure.axes[0].get_ylim())
        canvas = FigureCanvas(fig)
        vbox.pack_start(canvas)
        toolbar = NavigationToolbar(canvas, win)
        vbox.pack_start(toolbar, False, False)
        win.show_all()
        P.show()
    #-------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------
    def updateSlice(self):
        cset = self.sliceplot.setData(self.getSlice(), norm=self.norm)
        if self.overlay_data is not None:
            self.overlay.setData(self.getOverlaySlice(),norm=self.overlay_norm)
        if hasattr(self, "ROI") and self.ROI is not None:
            self.ROI.updateAx(self.sliceplot.getAxes())
        self.rowplot.setData(self.getRow())
        self.colplot.setData(self.getCol())
        self.status.cbar.setRange(self.sliceDataRange(), norm=self.norm)
        if self.contour_tools is not None:
            self.contour_tools.draw_bar(cset)

    #-------------------------------------------------------------------------
    def sliceDataRange(self):
        return self.getSlice().min(), self.getSlice().max()

    #------------------------------------------------------------------------- 
    def updateDataRange(self):
        data_min = self.transform(self.data[:]).min()
        data_max = self.transform(self.data[:]).max()
        self.rowplot.setDataRange(data_min, data_max)
        self.colplot.setDataRange(data_max, data_min)
        
    #-------------------------------------------------------------------------
    ### THIS IS HACKY!!
    def externalUpdate(self, new_img):
        self.img_obj = new_img
        self.data = SlicerImage(new_img)
        self.norm = None
        self.setNorm(contrast=self.contrast)
        self.updateDataRange()
        self.updateSlice()

    #-------------------------------------------------------------------------
    def radioHandler(self, button, transform):
        if not button.get_active(): return
        if self.orient_mode >= 0:
            mat = self.data.plane_xform(self.orient_mode)
            self.transform = compose_xform(mat, prefilter=transform)
        else:
            self.transform = transform
        self.updateDataRange()
        self.norm = None
        self.setNorm(contrast=self.contrast)
        self.updateSlice()

    #-------------------------------------------------------------------------
    def sliderHandler(self, adj):
        row_dim, col_dim = self.control_panel.slice_dims
        if adj.dim.index == row_dim: self.updateRow()
        elif adj.dim.index == col_dim: self.updateCol()
        else: self.updateSlice()

    #-------------------------------------------------------------------------
    def contrastHandler(self, adj):
        self.contrast = self.control_panel.getContrastLevel()
        self.setNorm(contrast=self.contrast)
        self.updateSlice()

    #-------------------------------------------------------------------------
    ###### Event handlers for sliceplot events
    def sliceResizeHandler(self, event):
        self.scale_image()
        
    #-------------------------------------------------------------------------
    def sliceMouseDownHandler(self, event):
        y, x = self.sliceplot.getEventCoords(event)
        self._dragging = True
        # make sure this registers as a "new" position
        self._mouse_x = self._mouse_y = None
        self.updateCoords(y,x)

    #-------------------------------------------------------------------------
    def sliceMouseUpHandler(self, event):
        y, x = self.sliceplot.getEventCoords(event)
        self._dragging = False

    #-------------------------------------------------------------------------
    def sliceMouseMotionHandler(self, event):
        y, x = self.sliceplot.getEventCoords(event)
        self.updateCoords(y,x)

    #-------------------------------------------------------------------------
    ###### Handlers for ROI button
    def ROIHandler(self, button):
        # when ROI button is pressed, turn off mouse-press events
        # (turn them back on after the rectangle-drawing handler is finished

        # if RS already drawn, do nothing
        if not (hasattr(self, "ROI") and self.ROI):
            self.status.toggle_button(self.clearROI)
            self.connectFigureCanvasEvents(mode="disable")
            self.sliceplot.toggleCrosshairs(mode=False)
            ax = self.sliceplot.getAxes()
            rect_props = dict(facecolor='red', edgecolor='cyan', alpha=0.25,
                              fill=True)
            self.ROI = RectangleSelector(ax, self.rectangleHandler,
                                         drawtype="box", useblit=True,
                                         rectprops=rect_props)
            self.sliceplot.draw()
            
    #-------------------------------------------------------------------------
    def rectangleHandler(self, event1, event2):
        x1,y1 = map(lambda x: int(round(x)), [event1.xdata, event1.ydata])
        x2,y2 = map(lambda x: int(round(x)), [event2.xdata, event2.ydata])
        (x1,x2,y1,y2) = (min(x1,x2),max(x1,x2),min(y1,y2),max(y1,y2))
        self.ROI.is_active(False)
        self.connectFigureCanvasEvents(mode="enable")
        self.sliceplot.toggleCrosshairs(mode=True)
        #self.ROI = None
        avg = self.getSlice()[y1:y2,x1:x2].mean()
        text = "average in [%d,%d] to [%d,%d]: %2.4f"%(x1,y1,x2-1,y2-1,avg)
        self.status.setROILabel(text)

    #-------------------------------------------------------------------------
    def clearROI(self, button):
        self.connectFigureCanvasEvents(mode="enable")
        self.sliceplot.toggleCrosshairs(mode=True)
        self.ROI.is_active(False)
        self.ROI.clear()
        self.ROI = None
        self.status.toggle_button(self.ROIHandler)
    
    ###### Helpers for object-to-object coordination
    def connectFigureCanvasEvents(self, mode="enable"):
        if mode == "init":
            # once enabled, these two never get disconnected
            self.sliceplot.mpl_connect(
                "motion_notify_event", self.sliceMouseMotionHandler)
            self.sliceplot.mpl_connect(
                "resize_event", self.sliceResizeHandler)
##             self.colplot.mpl_connect("button_press_event",
##                                      self.colplot.popup)
##             self.rowplot.mpl_connect("button_press_event",
##                                      self.rowplot.popup)
            self.colplot.mpl_connect("button_press_event", self.popup)
            self.rowplot.mpl_connect("button_press_event", self.popup)
            self.slice_canvas_mode = mode
            mode = "enable"
        if mode == "enable" and self.slice_canvas_mode != "enable":
            # mouse-pressing events may be disabled
            self.press_id = self.sliceplot.mpl_connect(
                "button_press_event", self.sliceMouseDownHandler)
            self.release_id = self.sliceplot.mpl_connect(
                "button_release_event", self.sliceMouseUpHandler)
            self.slice_canvas_mode = mode
        elif mode == "disable" and self.slice_canvas_mode != "disable":
            self.sliceplot.mpl_disconnect(self.press_id)
            self.sliceplot.mpl_disconnect(self.release_id)
            self.slice_canvas_mode = mode
    #-------------------------------------------------------------------------
    def updateCoords(self, y, x):

        # do nothing if coords haven't changed
        if x == self._mouse_x and y == self._mouse_y: return
        self._mouse_x, self._mouse_y = x, y

        # update statusbar element value label
        self.updateStatusLabel(y, x)

        # update crosshairs and projection plots if button down
        if self._dragging: self.updateProjections(y,x)

    #------------------------------------------------------------------------- 
    def updateProjections(self, y, x):
        "Update crosshairs and row and column plots."
        if x != None and y != None:
            self.control_panel.setRowIndex(y)
            self.control_panel.setColIndex(x)
            self.updateCrosshairs()

    #------------------------------------------------------------------------- 
    def updateCrosshairs(self):
        self.sliceplot.setCrosshairs(
          self.control_panel.getColIndex(),
          self.control_panel.getRowIndex())
        
    #------------------------------------------------------------------------- 
    def updateStatusLabel(self, y, x):
        if x != None and y != None:
            text = "[%d,%d] = %.4f"%(x, y, self.getSlice()[y,x])
        else: text = ""
        self.status.setLabel(text)

    #------------------------------------------------------------------------- 
    def setNorm(self, contrast=1.):
        scale = -0.75*(contrast-1.0) + 1.0
        if self.norm is None:
            # set the black point to 1st percentile value
            # set the white point to 99th percentile value
            sorted = N.sort(self.transform(self.data[:]).flatten())
            npts = sorted.shape[0]
            self.blkpt, self.whtpt = sorted[int(.01*npts+0.5)], \
                                     sorted[int(.99*npts+0.5)]

        if self.overlay_data is not None:
            if self.overlay_norm is None:
                sorted = N.sort(self.transform(self.overlay_data[:]).flatten())
                npts = sorted.shape[0]
                self.blkpt_o, self.whtpt_o = sorted[int(.1*npts+0.5)], \
                                             sorted[int(.99*npts+0.5)]
            self.overlay_norm = P.normalize(vmin=self.blkpt_o,
                                            vmax=self.whtpt_o*scale)

        self.norm = P.normalize(vmin=self.blkpt, vmax=self.whtpt*scale)
   
    #-------------------------------------------------------------------------
    ###### Handlers and helpers for menubar actions
    def launch_contour_tool(self, action):
        if self.contour_tools is not None:
            self.contour_tools.present()
        else:
            self.contour_tools = ContourToolWin(self.sliceplot, self)

    #-------------------------------------------------------------------------
    def launch_overlay_toolbox(self, action):
        if self.overlay_data is not None:
            if not hasattr(self, "overlay_tools") or not self.overlay_tools:
                self.overlay_tools = OverlayToolWin(self.overlay, self)
            else:
                self.overlay_tools.present()

    #-------------------------------------------------------------------------
    def save_png(self, action):
        # save a PNG of the current image and the current scaling
        fname = ask_fname(self, "Save image as...")
        if fname is None:
            return
        fname = fname.rsplit(".")[-1] == "png" and fname or fname+".png"
        im = self.sliceplot.getImage().make_image()
        im.flipud_out()
        im.write_png(fname)

    #-------------------------------------------------------------------------
    def save_png_montage(self, action):
        # make a montage PNG, for now make 5 slices to a row
        # make image with longest side 128 pix
        fname = ask_fname(self, "Save montage as...")
        if fname is None:
            return
        fname = fname.rsplit(".")[-1] == "png" and fname or fname+".png"
        px, py = self.slice_proportions()
        scale = 128./max(px, py)
        wpix, hpix = round(scale*px), round(scale*py)
        indices = range(self.data.ndim)
        ridx,cidx = (self.control_panel.getRowDim().index,
                     self.control_panel.getColDim().index)
        
        indices.remove(ridx)
        indices.remove(cidx)
        sidx = indices[-1]
        nslice = self.data.shape[sidx]

        cmap = self.sliceplot.getImage().cmap

        #col_buf = 20
        #row_buf = 50 - 30
        col_buf = 0
        row_buf = 0
        lr_buf = 0 #20
        b_buf = 0 #20
        #title_buf = 50
        ncol = 6 # hardwired for now
        nrow = int(nslice/ncol) + (nslice % ncol and 1 or 0)
        # get required height and width in pixels
        _ht = float(10 + nrow*(hpix + row_buf) + b_buf)
        _wd = float(2*lr_buf + ncol*wpix + (ncol-1)*col_buf)
        figdpi = 100
        # inches = _ht/dpi, _wd/dpi
        figsize = (_wd/figdpi, _ht/figdpi)
        fig = P.Figure(figsize=figsize, dpi=figdpi)
        fig.set_canvas(FigureCanvas(fig))
        plane_slice = list(self.control_panel.getIndexSlices())
        for row in range(nrow):
            for col in range(ncol):
                s = col + row*ncol
                if s >= nslice:
                    continue
                plane_slice[sidx] = s
                Loff = (lr_buf + (col)*(wpix + col_buf))/_wd
                Boff = (b_buf + (nrow-row-1)*(hpix + row_buf))/_ht
                Xpct, Ypct = (wpix/_wd, hpix/_ht)
                ax = fig.add_axes([Loff, Boff, Xpct, Ypct])
                ax.imshow(self.transform(self.data[tuple(plane_slice)]),
                          cmap=cmap,
                          origin='lower',
                          interpolation='nearest',
                          aspect='auto',
                          norm=self.norm)
                
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False)
                ax.set_frame_on(False)
                #t = ax.set_title('Slice %d'%s)
                #t.set_size(12)
        fig.savefig(fname, dpi=figdpi)

    #-------------------------------------------------------------------------
    def save_movie(self, axis):
        # axis will be either -4 (time) or -3 (slice)
        mov_filter = gtk.FileFilter()
        mov_filter.add_pattern("*.avi")
        mov_filter.set_name("Movie files")
        fname = ask_fname(self, "Save movie as...", filter=mov_filter)
        if fname is None:
            return
        fname = fname.rsplit(".")[-1] == "avi" and fname or fname+".avi"
        tmp_name = os.path.join(os.path.split(fname)[0], '_tmpimg')
        # set up the plot
        px, py = self.slice_proportions()
        scale = 512./max(px, py)
        wpix, hpix = round(scale*px), round(scale*py)
        cmap = self.sliceplot.getImage().cmap
        interp = self.sliceplot.getImage().get_interpolation()
        figdpi = 100.
        figsize = (wpix/figdpi, hpix/figdpi)
        fig = P.Figure(figsize=figsize, dpi=figdpi)
        fig.set_canvas(FigureCanvas(fig))
        ax = fig.add_subplot(111)
        ax.imshow(self.getSlice(), cmap=cmap, interpolation=interp,
                  origin='lower', norm=self.norm)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        img = ax.images[0]
        indices = range(self.data.ndim)
        ridx,cidx = (self.control_panel.getRowDim().index,
                     self.control_panel.getColDim().index)
        
        indices.remove(ridx)
        indices.remove(cidx)
        sl_idx = indices[-1]

        current_slice = list(self.control_panel.getIndexSlices())
        # if axis indicates time, cycle through index0 of current_slice
        # if axis indicates slice, cycle through sl_idx of current_slice
        if axis == -4:
            cycle_idx = 0
        else:
            cycle_idx = sl_idx

        files = []
        for n in xrange(self.data.shape[cycle_idx]):
            current_slice[cycle_idx] = n
            img.set_data(self.transform(self.data[current_slice]))
            img.write_png(tmp_name+'%3d.png'%n)
            files.append(tmp_name+'%3d.png'%n)

        path_str = 'mf://'+tmp_name+'*.png'
        os.system("mencoder %s -mf w=512:h=512:type=png:fps=10 -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o %s"%(path_str, fname))
        for f in files:
            os.remove(f)

        
    def save_slice_movie(self, action):
        self.save_movie(-3)

    def save_time_movie(self, action):
        self.save_movie(-4)

        

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
        img = readImage(fname, vrange=(0,0))
        self.overlay_data = img
        if self.data.shape[-3] != img.shape[-3]:
            print "slices don't match"
            return
        self.setNorm(contrast=self.contrast)
        self.overlay = OverLay(self.sliceplot, norm=self.overlay_norm)
        self.updateSlice()

    #-------------------------------------------------------------------------
    def killoverlay(self, action):
        if self.overlay_data is not None:
            self.overlay.removeSelf()
            if hasattr(self, "overlay_tools") and self.overlay_tools:
                self.overlay_tools.destroy()
                del self.overlay_tools
            del self.overlay
            del self.overlay_data
            self.overlay_data = None
            self.overlay_norm = None

    #-------------------------------------------------------------------------
    def cmap_handler(self, action, current):
        # cmap_lookup defined in this module
        cmap = cmap_lookup[current.get_current_value()]
        self.sliceplot.setCmap(cmap)
        self.status.cbar.setCmap(cmap)

    #-------------------------------------------------------------------------
    def interpo_handler(self, action, current):
        interp_method = interpo_lookup[current.get_current_value()]
        self.sliceplot.setInterpo(interp_method)

    #-------------------------------------------------------------------------
    def orient_handler(self, action, current):
        self.orient_mode = current.get_current_value()
        if self.iscomplex:
            for button in self.control_panel.radios:
                if button.get_active():
                    prefilter = button.transform
        else:
            prefilter = ident_xform
        if self.orient_mode >= 0:
            ax, cor, sag = self.data.slicing()
            trans_dims = {
                ax: [cor, sag],
                cor: [ax, sag],
                sag: [ax, cor],
            }.get(self.orient_mode)
            #offset = len(self.data.shape)            
            #new_zyx = map(lambda x: x-3, [self.orient_mode] + trans_dims)
            new_zyx = [x-3 for x in ([self.orient_mode] + trans_dims)]
            T = self.data.plane_xform(self.orient_mode)
            self.transform = compose_xform(T, prefilter=prefilter)
        else:
            new_zyx = [-3, -2, -1]
            self.transform = prefilter
        self.control_panel.ResetDims(new_zyx)
        self.scale_image()
        # might have to update the sliceplot figure's figsize
        self.updateSlice()
        self.updateRow()
        self.updateCol()
        
    #-------------------------------------------------------------------------
    def scale_handler(self, action, current):
        self.image_scale = current.get_current_value()
        self.scale_image()

    #-------------------------------------------------------------------------
    def slice_proportions(self):
        im = self.data
        dimsizes = im.dr * N.array(im.shape[-3:])
        r_idx, c_idx = [x-im.ndim
                        for x in (self.control_panel.getRowDim().index,
                                  self.control_panel.getColDim().index)]
        r = dimsizes[r_idx] / dimsizes[c_idx]
        base_size = float(max(im.shape[r_idx], im.shape[c_idx]))
        px, py = r > 1 and (base_size/r, base_size) or (base_size, r*base_size)
        return (px, py)

    #-------------------------------------------------------------------------
    def scale_image(self):
        px, py = self.slice_proportions()
        scale = self.image_scale
        canvas_size_x, canvas_size_y = self.sliceplot.get_size_request()
        canvas_size_real_x,canvas_size_real_y=self.sliceplot.get_width_height()
        new_img_size = (scale*px, scale*py)
        # If the new image requires a larger canvas, resize it.
        # Otherwise, make sure the canvas is at the default size
        if max(canvas_size_x,canvas_size_y) < max(*new_img_size) + 50:
            canvas_size_real_y = canvas_size_real_x = \
                    canvas_size_y = canvas_size_x = max(*new_img_size) + 50
        elif max(canvas_size_x,canvas_size_y) > 400 and max(*new_img_size)<350:
            canvas_size_x = canvas_size_y = 350
            canvas_size_real_x = canvas_size_real_y = 396
        ax = self.sliceplot.getAxes()
        w = new_img_size[0]/canvas_size_real_x
        h = new_img_size[1]/canvas_size_real_y
        l = 15./canvas_size_real_x
        b = 1.0 - (new_img_size[1] + 25.)/canvas_size_real_y
        ax.set_position([l,b,w,h])
        self.sliceplot.set_size_request(int(canvas_size_x),int(canvas_size_y))
        self.sliceplot.draw()

    #-------------------------------------------------------------------------
    def launch_ortho_plots(self, action):
        from recon.visualization.spmclone import spmclone
        spmclone(self.data, parent=self)
    #-------------------------------------------------------------------------
    def launch_recon_gui(self, action):
        from recon.visualization.recon_gui import recon_gui
        recon_gui(image=self.img_obj, parent=self)

    #-------------------------------------------------------------------------
    def auto_scale_image(self):
        # try to find some scale that gets ~ 256x256 pixels
        base_img_size = self.slice_proportions()

        p = round(256./max(*base_img_size))
        self.image_scale = p
        self.scale_image()
        
##         r = dimsizes[r_idx] / dimsizes[c_idx]
##         px, py = r > 1 and (1/r, 1.0) or (1.0, r)
##         new_img_size =  max([int(p*x) for x in base_img_size])
##         canvas_size = 350
##         canvas_size_real = 396
##         ax = self.sliceplot.getAxes()
##         w = px*new_img_size/canvas_size_real
##         h = py*new_img_size/canvas_size_real
##         l = 15./canvas_size
##         b = 1.0 - (new_img_size + 25.)/canvas_size_real
##         ax.set_position([l,b,w,h])
##         self.sliceplot.set_size_request(canvas_size,canvas_size)
##         self.image_scale = p

    #-------------------------------------------------------------------------
    def _create_action_group(self, default_scale):
        entries = (
            ( "FileMenu", None, "_File" ),
            ( "MoviesMenu", None, "Save movies" ),
            ( "ToolsMenu", None, "_Tools" ),
            ( "SizeMenu", None, "_Image Size" ),
            ( "ColorMapping", None, "_Color Mapping"),
            ( "Interpolation", None, "_Interpolation"),
            ( "Orientation", None, "_Orientation" ),
            ( "Save Image", gtk.STOCK_SAVE,
              "_Save Image", "<control>S",
              "Saves current slice as PNG",
              self.save_png ),
            ( "Save Montage", gtk.STOCK_SAVE,
              "_Save Montage", "<control><shift>S",
              "Saves all slices as a montage",
              self.save_png_montage ),
            ( "Slice movie", None, "Slice movie", None,
              "Saves a movie scanning through slices",
              self.save_slice_movie),
            ( "Time movie", None, "Time movie", None,
              "Saves a movie scanning through time points",
              self.save_time_movie ),
            ( "Quit", gtk.STOCK_QUIT,
              "_Quit", "<control>Q",
              "Quits",
              lambda action: self.destroy() ),
            ( "Contour Plot", None,
              "_Contour Plot", None,
              "Opens contour plot controls",
              self.launch_contour_tool ),
            ( "View Ortho Plots", None,
              "_View Ortho Plots", None,
              "Opens orthogonal plot viewer",
              self.launch_ortho_plots ),
            ( "Run Recon GUI", None,
              "_Run Recon GUI", None,
              "", self.launch_recon_gui ),
            ( "OverlayMenu", None, "_Overlay" ),
            ( "Load Overlay", None,
              "_Load Overlay", "",
              "Load an image to overlay",
              self.initoverlay ),
            ( "Unload Overlay", None,
              "_Unload Overlay", "",
              "Unload the overlay",
              self.killoverlay ),
            ( "Overlay Adjustment Toolbox", None,
              "_Overlay Adjustment Toolbox", "",
              "Launch overlay toolbox",
              self.launch_overlay_toolbox ),
            
        )

        size_toggles = (
            ( "1x", None, "_1x", None, "", 1 ),
            ( "2x", None, "_2x", None, "", 2 ),
            ( "4x", None, "_4x", None, "", 4 ),
            ( "6x", None, "_6x", None, "", 6 ),
            ( "8x", None, "_8x", None, "", 8 )
        )

        orient_toggles = (
            ( "Native", None, "_Navive", None, "", -1 ),
            ( "Axial", None, "_Axial", None, "", self.data.ax ),
            ( "Coronal", None, "_Coronal", None, "", self.data.cor ),
            ( "Saggital", None, "_Saggital", None, "", self.data.sag )
        )

        cmap_toggles = tuple([(cmap.name.strip(), None,
                               "_"+cmap.name.strip(), None, "", num)
                              for num, cmap in cmap_lookup.items()])

        interpo_toggles = tuple([(i, None, "_"+i, None, "", num)
                                 for num, i in interpo_lookup.items()])

        action_group = gtk.ActionGroup("WindowActions")
        action_group.add_actions(entries)
        action_group.add_radio_actions(size_toggles, int(default_scale),
                                       self.scale_handler)
        action_group.add_radio_actions(cmap_toggles, 8,
                                       self.cmap_handler)
        action_group.add_radio_actions(interpo_toggles, 0,
                                       self.interpo_handler)
        action_group.add_radio_actions(orient_toggles, -1, self.orient_handler)
        if not self.img_obj:
            action_group.get_action("Run Recon GUI").set_sensitive(False)
        if self.data.ndim < 4:
            action_group.get_action("Time movie").set_sensitive(False)
        return action_group

##############################################################################
class ContourToolWin (gtk.Window):
    "A Window class defining a pop-up control widget for the contour plot."
    
    def __init__(self, obs_slice, parent):
        self.padre = parent
        self.sliceplot = obs_slice
        self.hbox = gtk.HBox(spacing=5)
        self.levSlider = gtk.VScale(gtk.Adjustment(7, 2, 20, 1, 1))
        self.levSlider.set_digits(0)
        self.levSlider.set_value_pos(gtk.POS_TOP)
        self.levSlider.get_adjustment().connect("value-changed",
                                                self.clevel_handler)
        self.hbox.pack_start(self.levSlider)
        self.fig = P.Figure(figsize=(1,4), dpi=80)
        self.cbar_ax = self.fig.add_axes([.1, .04, .45, .9])
        self.figcanvas = FigureCanvas(self.fig)
        self.figcanvas.set_size_request(100,50*4)
        self.hbox.pack_start(self.figcanvas)
        self.setContours(int(self.levSlider.get_value()))
        gtk.Window.__init__(self)
        self.set_destroy_with_parent(True)
        self.connect("destroy", self._takedown)
        self.set_default_size(150,400)
        self.set_title("Contour Plot Controls")
        self.set_border_width(3)
        self.add(self.hbox)
        self.show_all()
        P.show()
        #gtk.main()

    #-------------------------------------------------------------------------
    def _takedown(self, foo):
        self.sliceplot.killContour()
        self.padre.contour_tools = None
        foo.destroy()

    #-------------------------------------------------------------------------
    def setContours(self, levels):
        cset = self.sliceplot.doContours(levels)
        self.draw_bar(cset)

    #-------------------------------------------------------------------------
    def draw_bar(self, cset):
        # try to fix cset levels to 4 significant digits
        cset.levels = N.floor(0.5 + cset.levels*1000)/1000.
        self.cbar_ax.clear()
        self.fig.colorbar(cset, self.cbar_ax)
        self.figcanvas.draw()        

    #-------------------------------------------------------------------------
    def clevel_handler(self, adj):
        self.setContours(int(self.levSlider.get_value()))

##############################################################################
class OverlayToolWin (gtk.Window):
    "A Window class defining a pop-up control widget for the overlay plot."

    def __init__(self, overlay_obj, parent):
        self.padre = parent
        self.overlay_ref = overlay_obj
        self.vbox = gtk.VBox(spacing=5)
        self.vbox.set_border_width(10)        
        alpha = self.overlay_ref.alpha

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
        for interpo in interpo_lookup.values():
            self.interpo_list.append_text(interpo)
        self.interpo_list.set_active(interpo_lookup.values().index("nearest"))
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
        self.overlay_ref.setAlpha(self.alphaslider.get_value())

    #-------------------------------------------------------------------------
    def cmap_handler(self, cbox):
        cmap = cmap_lookup[cbox.get_active()]
        self.overlay_ref.setCmap(cmap)

    #-------------------------------------------------------------------------
    def interpo_handler(self, cbox):
        interpo = interpo_lookup[cbox.get_active()]
        self.overlay_ref.setInterpo(interpo)

##############################################################################
class Dimension (object):
    "a Dimension has an index in the dim ordering, a total size, and a name"
    def __init__(self, index, size, name):
        self.index = index
        self.size = size
        self.name = name


##############################################################################
class DimSlider (gtk.HScale):
    def __init__(self, dim):
        adj = gtk.Adjustment(0, 0, dim.size-1, 1, 1)
        adj.dim = dim
        gtk.HScale.__init__(self, adj)
        self.set_digits(0)
        self.set_value_pos(gtk.POS_RIGHT)
        self.handler = None

    def changeDim(self, dim, pt):
        adj = self.get_adjustment()
        if self.handler:
            adj.handler_block(self.handler)
        self.set_range(0, dim.size-1)
        adj.upper = dim.size-1
        adj.set_value(pt)
        adj.dim = dim
        if self.handler:
            adj.handler_unblock(self.handler)
        
    

##############################################################################
class ContrastSlider (gtk.HScale):
    def __init__(self):
        gtk.HScale.__init__(self, gtk.Adjustment(1.0, 0.05, 2.0, 0.025, 0.05))
        self.set_digits(2)
        self.set_value_pos(gtk.POS_RIGHT)


##############################################################################
class ControlPanel (gtk.Frame):
    "A Frame class containing dimension slider widgets and button widgets"
    #-------------------------------------------------------------------------
    def __init__(self, shape, dim_names=[], iscomplex=False):
        self._init_dimensions(shape, dim_names)
        gtk.Frame.__init__(self)
        main_vbox = gtk.VBox()
        main_vbox.set_border_width(2)

        # radio buttons for different aspects of complex data
        xform_map = {
          "ident": ident_xform,
          "abs": abs_xform,
          "phs": phs_xform,
          "real": real_xform,
          "imag": imag_xform}
        self.radios = []
        radio_box = gtk.HBox()
        prev_button = None
        for name in ("abs","phs","real","imag"):
            button = prev_button = gtk.RadioButton(prev_button, name)
            button.transform = xform_map[name]
            if name=="abs": button.set_active(True)
            self.radios.append(button)
            radio_box.add(button)
        if iscomplex:
            main_vbox.pack_end(radio_box, False, False, 0)
            main_vbox.pack_end(gtk.HSeparator(), False, False, 0)

        # slider for each data dimension
        self.sliders = [DimSlider(dim) for dim in self.dimensions]
        for slider, dim in zip(self.sliders, self.dimensions):
            self._add_slider(slider, "%s:"%dim.name, main_vbox)

        # start with the center row and column
        rowdim = self.getRowDim()
        self.sliders[rowdim.index].set_value(rowdim.size/2)
        coldim = self.getColDim()
        self.sliders[coldim.index].set_value(coldim.size/2)

        # slider for contrast adjustment
        self.contrast_slider = ContrastSlider()
        self._add_slider(self.contrast_slider, "Contrast:", main_vbox)

        self.add(main_vbox)

    #-------------------------------------------------------------------------
    def _add_slider(self, slider, label, vbox):
        label = gtk.Label(label)
        label.set_alignment(0, 0.5)
        vbox.pack_start(label, False, False, 0)
        vbox.pack_start(slider, False, False, 0)


    #-------------------------------------------------------------------------
    def _init_dimensions(self, dim_sizes, dim_names):
        self.dimensions = []
        num_dims = len(dim_sizes)
        num_names = len(dim_names)
        if num_names != num_dims:
            dim_names = ["Dim %s"%i for i in range(num_dims)]
        for dim_num, (dim_size, dim_name) in\
          enumerate(zip(dim_sizes, dim_names)):
            self.dimensions.append( Dimension(dim_num, dim_size, dim_name) )
        # initialize everything with native slicing, this may change later
        self.slice_dims = (self.dimensions[-2].index, self.dimensions[-1].index)
        self.slicing = range(len(self.dimensions))

    #-------------------------------------------------------------------------
    def connect(self, radio_handler, slider_handler, contrast_handler):
        "Connect control elements to the given handler functions."

        # connect radio buttons
        for r in self.radios: r.connect("toggled", radio_handler, r.transform)

        # connect slice position sliders
        for s in self.sliders:
            id = s.get_adjustment().connect("value_changed", slider_handler)
            s.handler = id

        # connect contrast slider
        self.contrast_slider.get_adjustment().connect(
          "value_changed", contrast_handler)

    #-------------------------------------------------------------------------
    def getContrastLevel(self):
        return self.contrast_slider.get_adjustment().value

    #-------------------------------------------------------------------------
    def getDimPosition(self, dnum):
        dim_slider = self.slicing[dnum]
        return int(self.sliders[dim_slider].get_adjustment().value)

    #-------------------------------------------------------------------------
    def setDimPosition(self, dnum, index):
        dim_slider = self.slicing[dnum]
        return self.sliders[dim_slider].get_adjustment().set_value(int(index))

    #-------------------------------------------------------------------------
    def getRowIndex(self): return self.getDimPosition(self.slice_dims[-2])

    #-------------------------------------------------------------------------
    def getColIndex(self): return self.getDimPosition(self.slice_dims[-1])

    #------------------------------------------------------------------------- 
    def setRowIndex(self, index): self.setDimPosition(self.slice_dims[-2],index)

    #------------------------------------------------------------------------- 
    def setColIndex(self, index): self.setDimPosition(self.slice_dims[-1],index)

    #------------------------------------------------------------------------- 
    def getRowDim(self): return self.dimensions[self.slice_dims[-2]]

    #------------------------------------------------------------------------- 
    def getColDim(self): return self.dimensions[self.slice_dims[-1]]

    #-------------------------------------------------------------------------
    def ResetDims(self, dim_order):
        # tasks:
        # assign new dim references to each slider (except 0th if tdim)
        # identify new transverse slicing directions (rowdim, coldim)
        #
        # define new slice_dims according to new slice plane
        # reset ranges set_range(min, max) for the reordered sliders
        # set current values set_value(p) of reordered sliders
        for n, slider in enumerate(self.sliders[-3:]):
            new_dim = self.dimensions[dim_order[n]]
            if n:
                initial_value = new_dim.size/2
            else:
                initial_value = 0
            slider.changeDim(new_dim, initial_value)

        self.slice_dims = (self.dimensions[dim_order[-2]].index,
                           self.dimensions[dim_order[-1]].index)
        if len(self.dimensions) > 3:
            offset = 1
        else:
            offset = 0
        self.slicing = [dim_order.index(n)+offset for n in range(-3,0,1)]
        if offset: self.slicing = [0] + self.slicing
    #-------------------------------------------------------------------------
    def getIndexSlices(self):
        return tuple([
            dim.index in self.slice_dims and slice(None) or \
            self.getDimPosition(dim.index) for dim in self.dimensions])


##############################################################################
class RowPlot (FigureCanvas):
    "A Canvas class containing a matplotlib plot"
    #-------------------------------------------------------------------------
    def __init__(self, data):
        fig = P.Figure(figsize=(3., 6.))
        ax  = fig.add_axes([0.05, 0.05, 0.85, 0.85])
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        FigureCanvas.__init__(self, fig)
        self.setData(data)

    #-------------------------------------------------------------------------
    def setDataRange(self, data_min, data_max):
        self.figure.axes[0].set_ylim(data_min, data_max)

    #-------------------------------------------------------------------------
    def setData(self, data):
        ax = self.figure.axes[0]
        indices = N.arange(len(data))
        if not hasattr(self, "data"): ax.plot(indices, data)
        else: ax.lines[0].set_data(indices, data)
        ax.set_xlim(-.5, len(data)-.5)
        self.data = data
        self.draw()

    #------------------------------------------------------------------------- 
    def popup(self, event):
        win = gtk.Window()
        win.connect("destroy", lambda x: x.destroy())
        win.set_default_size(400,300)
        vbox = gtk.VBox()
        win.add(vbox)
        fig = P.Figure()
        ax = fig.add_subplot(111)
        ax.plot(self.data)
        ax.set_xlim(0, self.data.shape[0])
        ax.set_ylim(*self.figure.axes[0].get_ylim())
        canvas = FigureCanvas(fig)
        vbox.pack_start(canvas)
        toolbar = NavigationToolbar(canvas, win)
        vbox.pack_start(toolbar, False, False)
        win.show_all()
        P.show()
        #gtk.main()


##############################################################################
class ColPlot (FigureCanvas):
    "A Canvas class containing a matplotlib plot"    
    #-------------------------------------------------------------------------
    def __init__(self, data):
        fig = P.Figure(figsize=(6., 3.))
        fig.add_axes([0.1, 0.1, 0.85, 0.85])
        FigureCanvas.__init__(self, fig)
        self.setData(data)

    #-------------------------------------------------------------------------
    def setDataRange(self, data_min, data_max):
        self.figure.axes[0].set_xlim(data_min, data_max)

    #-------------------------------------------------------------------------
    def setData(self, data):
        ax = self.figure.axes[0]
        indices = N.arange(len(data))
        if not hasattr(self, "data"): ax.plot(data, indices)
        else: ax.lines[0].set_data(data, indices)
        ax.set_ylim(-.5,len(data)-.5)
        self.data = data
        self.draw()

    #-------------------------------------------------------------------------
    def popup(self, event):
        win = gtk.Window()
        win.connect("destroy", lambda x: x.destroy())
        win.set_default_size(400,300)
        vbox = gtk.VBox()
        win.add(vbox)
        fig = P.Figure()
        ax = fig.add_subplot(111)
        ax.plot(self.data)
        ax.set_xlim(0, self.data.shape[0])
        # in this plot the true y-axis runs backwards along the x-axis
        y_lim = self.figure.axes[0].get_xlim()
        ax.set_ylim(y_lim[::-1])
        canvas = FigureCanvas(fig)
        vbox.pack_start(canvas)
        toolbar = NavigationToolbar(canvas, win)
        vbox.pack_start(toolbar, False, False)
        win.show_all()
        P.show()
        #gtk.main()
        
##############################################################################
class SlicePlot (FigureCanvas):
    "A Canvas class containing a 2D matplotlib plot"    
    #-------------------------------------------------------------------------
    def __init__(self, data, x, y, cmap=P.cm.bone,
                 norm=None, interpolation="nearest"):
        self.norm = norm
        self.cmap = cmap        
        self.interpolation=interpolation
        self.hasContours = False
        self.contourLevels = 7
        fig = P.Figure(figsize=P.figaspect(data), dpi=80)
        ax = fig.add_subplot(111)
        ax.yaxis.tick_right()
        ax.title.set_y(1.05) 
        FigureCanvas.__init__(self, fig)
        self.setData(data)
        self._init_crosshairs(x, y)

    #-------------------------------------------------------------------------
    def _init_crosshairs(self, x, y):
        row_data, col_data = self._crosshairs_data(x, y)
        row_line = Line2D(row_data[0], row_data[1], color="r", alpha=.5)
        col_line = Line2D(col_data[0], col_data[1], color="r", alpha=.5)
        self.crosshairs = (row_line, col_line)
        ax = self.getAxes()
        ax.add_artist(row_line)
        ax.add_artist(col_line)

    #-------------------------------------------------------------------------
    def _crosshairs_data(self, x, y):
        data_height, data_width = self.data.shape
        row_data = ((x+.5-data_width/4., x+.5+data_width/4.), (y+.5, y+.5))
        col_data = ((x+.5, x+.5), (y+.5-data_height/4., y+.5+data_height/4.))
        return row_data, col_data

    #------------------------------------------------------------------------- 
    def setCrosshairs(self, x, y):
        row_data, col_data = self._crosshairs_data(x, y)
        row_line, col_line = self.crosshairs
        row_line.set_data(*row_data)
        col_line.set_data(*col_data)
        self.draw()

    #-------------------------------------------------------------------------
    def toggleCrosshairs(self, mode=True):
        for line in self.crosshairs:
            line.set_visible(mode)
        self.draw()

    #-------------------------------------------------------------------------
    def getEventCoords(self, event):
        if event.xdata is not None: x = int(event.xdata)
        else: x = None
        if event.ydata is not None:y = int(event.ydata)
        else: y = None
        if x < 0 or x >= self.data.shape[1]: x = None
        if y < 0 or y >= self.data.shape[0]: y = None
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
    def setCmap(self, cmapObj):
        self.setData(self.data, cmap=cmapObj)
    #-------------------------------------------------------------------------
    def setInterpo(self, interp_method):
        self.setData(self.data, interpolation=interp_method)
        
    #-------------------------------------------------------------------------
    def setData(self, data, norm=None, cmap=None, interpolation=None):
        ax = self.getAxes()
        if interpolation: self.interpolation = interpolation
        if cmap: self.cmap = cmap
        if norm: self.norm = norm
        try:
            img = self.getImage()
            img.set_data(data)
        except:
            ax.imshow(data, origin="lower", aspect="auto")
            img = self.getImage()
        
        img.set_cmap(self.cmap)
        img.set_interpolation(self.interpolation)
        img.set_norm(self.norm)
        nrows, ncols = data.shape[:2]
        ax.set_xlim((0,ncols))
        ax.set_ylim((0,nrows))
        self.data = data
        if self.hasContours:
            return self.doContours(self.contourLevels)
        else:
            self.draw()
            return None

    #-------------------------------------------------------------------------
    def doContours(self, levels):
        self.hasContours = True
        self.contourLevels = levels
        ax = self.getAxes()
        ax.collections = []
        mn, mx = self.data.min(), self.data.max()
        mx = mx + (mx-mn)*.001
        intv = matplotlib.transforms.Interval(
            matplotlib.transforms.Value(mn),
            matplotlib.transforms.Value(mx))
        #locator = matplotlib.ticker.MaxNLocator(levels+1)
        locator = matplotlib.ticker.LinearLocator(levels+1)
        locator.set_view_interval(intv)
        locator.set_data_interval(intv)
        clevels = locator()[:levels]
        if 0 in clevels: clevels[P.find(clevels==0)[0]] = 10.0
        cset = ax.contour(self.data, clevels, origin='lower', cmap=P.cm.hot)
        self.draw()
        return cset

    #-------------------------------------------------------------------------
    def killContour(self):
        ax = self.getAxes()
        ax.collections = []
        self.hasContours = False
        self.draw()
    
##############################################################################
class OverLay (object):
    """
    An Overlay inherits from the SlicePlot, and shares a Figure object
    with an existing SlicePlot.
    """

    def __init__(self, sliceplot, alpha=.55, cmap=P.cm.gist_heat,
                 norm=None, interpolation="nearest"):

        self.sliceplot = sliceplot
        self.norm = norm
        self.cmap = cmap
        self.alpha = alpha
        self.interpolation = interpolation
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
        if self.sliceplot.getImage(num=1) is None:
            ax.imshow(data, origin="lower")

        # retrieve overlay image ref
        img = self.sliceplot.getImage(num=1)
        img.set_data(data)
        img.set_cmap(self.cmap)
        img.set_alpha(self.alpha)
        img.set_interpolation(self.interpolation)
        img.set_norm(self.norm)
        self.data = data
        self.sliceplot.draw()

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
        if interpo in interpo_lookup.values():
            self.interpolation = interpo
            self.setData(self.data)
    #-------------------------------------------------------------------------
    def removeSelf(self):
        ax = self.sliceplot.getAxes()
        if len(ax.images) > 1:
            ax.images.pop(1)
            self.sliceplot.draw()


##############################################################################
class ColorBar (FigureCanvas):
    "A Canvas class showing the constrast scaling"
    #-------------------------------------------------------------------------
    def __init__(self, range, cmap=P.cm.bone, norm=None):
        fig = P.Figure(figsize = (16.0,2.0))
        fig.add_axes((0.05, 0.4, 0.9, 0.3))
        FigureCanvas.__init__(self, fig)
        self.figure.axes[0].yaxis.set_visible(False)
        self.cmap = cmap
        self.draw()
        self.setRange(range, norm=norm)

    #-------------------------------------------------------------------------
    def setCmap(self, cmapObj):
        self.cmap = cmapObj
        self.setRange(self.range, self.norm)
    #-------------------------------------------------------------------------
    def setRange(self, range, norm=None):
        self.norm = norm
        self.range = dMin, dMax = range
        ax = self.figure.axes[0]

        if dMin == dMax:
            # matplotlib is going to cry about this case anyway
            r_pts = N.zeros((128,))
            tx = N.asarray([0])
        else:
            r_pts = N.linspace(dMin, dMax, 128)
            tx = N.linspace(dMin, dMax, 7)
            # truncate to 4 digits
            tx = N.floor(0.5 + 1000*tx)/1000.
        data = N.outer(N.ones(5),r_pts)
        # need to clear axes because axis Intervals weren't updating
        ax.clear()
        ax.imshow(data, interpolation="nearest",
                  cmap=self.cmap, norm=norm,
                  extent=(r_pts[0], r_pts[-1], 0, 1), aspect="auto")
        #ax.images[0].set_data(data)
        ax.xaxis.set_ticks(tx)
        for tk in ax.xaxis.get_ticklabels(): tk.set_size(10.0)
        self.data = data
        self.draw()


##############################################################################
class StatusBar (gtk.Frame):

    DRAW_MODE = 0
    CLEAR_MODE = 1
    
    def __init__(self, sliceplot, range, cmap, norm):
        gtk.Frame.__init__(self)
        self.set_border_width(3)
        vbox = gtk.VBox(spacing=5)
        upper_hbox = gtk.HBox()
        upper_hbox.set_border_width(0)
        lower_hbox = gtk.HBox()
        lower_hbox.set_border_width(0)
        
        # colorbar
        self.cbar = ColorBar(range, cmap=cmap, norm=norm)
        self.cbar.set_size_request(400,20)
        upper_hbox.add(self.cbar)
 
        # pixel value
        self.label = gtk.Label()
        self.label.set_alignment(1, 0.5)
        self.label.set_size_request(140,20)
        self.label.set_line_wrap(True)
        upper_hbox.add(self.label)

        # rectangle-draw button
        self.drawbutton = gtk.Button(label="Get ROI average")
        #self.drawbutton.connect("clicked", self.button_handler)
        self.drawbutton.set_size_request(200,20)
        lower_hbox.add(self.drawbutton)
        self.button_mode = StatusBar.DRAW_MODE

        ## rectangle clear button
        #self.clearbutton = gtk.Button(label="Clear ROI")

        # average report
        self.roi_label = gtk.Label()
        self.roi_label.set_justify(gtk.JUSTIFY_CENTER)
        self.roi_label.set_line_wrap(True)
        self.roi_label.set_size_request(400,20)
        lower_hbox.add(self.roi_label)

        vbox.add(lower_hbox)
        vbox.add(upper_hbox)
        self.add(vbox)
        self.show_all()
        self.sliceplot_ref = sliceplot

    #-------------------------------------------------------------------------
    def connect_button(self, button_handler):
        self.handler_id = self.drawbutton.connect("clicked", button_handler)
    
    #-------------------------------------------------------------------------
    def toggle_button(self, new_handler):
        if self.button_mode == StatusBar.DRAW_MODE:
            self.drawbutton.set_label("clear ROI")
        else:
            self.drawbutton.set_label("Get ROI average")
        self.drawbutton.disconnect(self.handler_id)
        self.handler_id = self.drawbutton.connect("clicked", new_handler)
        self.drawbutton.show()
        self.button_mode = self.button_mode ^ 1

    #-------------------------------------------------------------------------
    def setLabel(self, text):
        self.label.set_text(text)
        
    #-------------------------------------------------------------------------
    def setROILabel(self, text):
        self.roi_label.set_text(text)

##############################################################################
##############################################################################
############################ MATPLOTLIB HACK #################################
class RectangleSelector:
    """
    Select a min/max range of the x axes for a matplotlib Axes

    Example usage:

      ax = subplot(111)
      ax.plot(x,y)

      def onselect(eclick, erelease):
          'eclick and erelease are matplotlib events at press and release'
          print 'startposition : (%f,%f)'%(eclick.xdata, eclick.ydata)
          print 'endposition   : (%f,%f)'%(erelease.xdata, erelease.ydata)
          print 'used button   : ', eclick.button

      span = Selector(ax, onselect,drawtype='box')
      show()

    """
    def __init__(self, ax, onselect, drawtype='box',
                 minspanx=None, minspany=None, useblit=False,
                 lineprops=None, rectprops=None):

        """
        Create a selector in ax.  When a selection is made, clear
        the span and call onselect with

          onselect(pos_1, pos_2)

        and clear the drawn box/line. There pos_i are arrays of length 2
        containing the x- and y-coordinate.

        If minspanx is not None then events smaller than minspanx
        in x direction are ignored(it's the same for y).

        The rect is drawn with rectprops; default
          rectprops = dict(facecolor='red', edgecolor = 'black',
                           alpha=0.5, fill=False)

        The line is drawn with lineprops; default
          lineprops = dict(color='black', linestyle='-',
                           linewidth = 2, alpha=0.5)

        Use type if you want the mouse to draw a line, a box or nothing
        between click and actual position ny setting
        drawtype = 'line', drawtype='box' or drawtype = 'none'.


        """
        self.ax = ax
        self.visible = True
        self.canvas = ax.figure.canvas
        self.connect_id = []
        self.active = True
        self.is_active(True)
##         self.canvas.mpl_connect('motion_notify_event', self.onmove)
##         self.canvas.mpl_connect('button_press_event', self.press)
##         self.canvas.mpl_connect('button_release_event', self.release)
##         self.canvas.mpl_connect('draw_event', self.update_background)

        self.to_draw = None
        self.background = None

        if drawtype == 'none':
            drawtype = 'line'                        # draw a line but make it
            self.visible = False                     # invisible

        if drawtype == 'box':
            if rectprops is None:
                rectprops = dict(facecolor='white', edgecolor = 'black',
                                 alpha=0.5, fill=False)
            self.rectprops = rectprops
            self.to_draw = Rectangle((0,0), 0, 1,visible=False,**self.rectprops)
            self.ax.add_patch(self.to_draw)
##         if drawtype == 'line':
##             if lineprops is None:
##                 lineprops = dict(color='black', linestyle='-',
##                                  linewidth = 2, alpha=0.5)
##             self.lineprops = lineprops
##             self.to_draw = Line2D([0,0],[0,0],visible=False,**self.lineprops)
##             self.ax.add_line(self.to_draw)

        self.onselect = onselect
        self.useblit = useblit
        self.minspanx = minspanx
        self.minspany = minspany
        self.drawtype = drawtype
        # will save the data (position at mouseclick)
        self.lastpress = self.eventpress = None
        # will save the data (pos. at mouserelease)
        self.lastrelease = self.eventrelease = None

    def update_background(self, event):
        'force an update of the background'
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)


    def is_active(self, active):
        """ Use this to activate the RectangleSelector from your program.
        """
        self.active = active
        if active and len(self.connect_id) == 0: # you want to activate and it
                                                   #  isn't already active
            self.connect_id.append(self.canvas.mpl_connect(
                  'motion_notify_event', self.onmove))
            self.connect_id.append(self.canvas.mpl_connect(
                  'button_press_event', self.press))
            self.connect_id.append(self.canvas.mpl_connect(
                  'button_release_event', self.release))
            self.connect_id.append(self.canvas.mpl_connect(
                  'draw_event', self.update_background))

        if not active and len(self.connect_id) != 0:  # you want to deactivate
            for index in self.connect_id:     #  and it isn't already inactive
                self.canvas.mpl_disconnect(index)
            self.connect_id = []

    def ignore(self, event):
        'return True if event should be ignored'
        # If no button was pressed yet ignore the event if it was out
        # of the axes
        if self.eventpress == None:
            return event.inaxes!= self.ax

        # If a button was pressed, check if the release-button is the
        # same.
        return  (event.inaxes!=self.ax or
                 event.button != self.eventpress.button)

    def updateAx(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        if self.to_draw not in ax.patches:
            ax.add_patch(self.to_draw)
        self.onselect(self.lastpress, self.lastrelease)
        self.canvas.draw()

    def clear(self):
        self.to_draw.set_visible(False)
        self.canvas.draw()

    def press(self, event):
        'on button press event'
        # Is the correct button pressed within the correct axes?
        if self.ignore(event): return


        # make the drawed box/line visible get the click-coordinates,
        # button, ...
        self.to_draw.set_visible(self.visible)
        self.eventpress = event
        return False


    def release(self, event):
        'on button release event'
        if self.eventpress is None or self.ignore(event): return
        # make the box/line invisible again
        #self.to_draw.set_visible(False)
        #self.canvas.draw()
        # release coordinates, button, ...
        self.eventrelease = event
        xmin, ymin = self.eventpress.xdata, self.eventpress.ydata
        xmax, ymax = self.eventrelease.xdata, self.eventrelease.ydata
        # calculate dimensions of box or line get values in the right
        # order
        if xmin>xmax: xmin, xmax = xmax, xmin
        if ymin>ymax: ymin, ymax = ymax, ymin



        spanx = xmax - xmin
        spany = ymax - ymin
        xproblems = self.minspanx is not None and spanx<self.minspanx
        yproblems = self.minspany is not None and spany<self.minspany
        if (self.drawtype=='box')  and (xproblems or  yproblems):
            """Box to small"""     # check if drawed distance (if it exists) is
            return                 # not to small in neither x nor y-direction
        if (self.drawtype=='line') and (xproblems and yproblems):
            """Line to small"""    # check if drawed distance (if it exists) is
            return                 # not to small in neither x nor y-direction
        self.onselect(self.eventpress, self.eventrelease)
                                              # call desired function
        self.lastpress = self.eventpress
        self.lastrelease = self.eventrelease
        self.eventpress = None                # reset the variables to their
        self.eventrelease = None              #   inital values
        return False

    def update(self):
        'draw using newfangled blit or oldfangled draw depending on useblit'
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.to_draw)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False


    def onmove(self, event):
        'on motion notify event if box/line is wanted'
        if self.eventpress is None or self.ignore(event): return
        x,y = event.xdata, event.ydata            # actual position (with
                                                  #   (button still pressed)
        if self.drawtype == 'box':
            minx, maxx = map(round, [self.eventpress.xdata, x]) # click-x and actual mouse-x
            miny, maxy = map(round, [self.eventpress.ydata, y]) # click-y and actual mouse-y
            if minx>maxx: minx, maxx = maxx, minx # get them in the right order
            if miny>maxy: miny, maxy = maxy, miny
            self.to_draw.xy[0] = minx             # set lower left of box
            self.to_draw.xy[1] = miny
            self.to_draw.set_width(maxx-minx)     # set width and height of box
            self.to_draw.set_height(maxy-miny)
            self.update()
            return False
        if self.drawtype == 'line':
            self.to_draw.set_data([self.eventpress.xdata, x],
                                  [self.eventpress.ydata, y])
            self.update()
            return False


##############################################################################
##############################################################################
##############################################################################


        
if __name__ == "__main__":
    from pylab import randn
    import pdb
    #pdb.run('sliceview(randn(6,6))', globals=globals(), locals=locals())
    pdb.run('sliceview(img.data)', globals=globals(), locals=locals())
    #sliceview(img.data.data)
