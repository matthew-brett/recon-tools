"The sliceview module defines classes providing a slice-plotting GUI"

import gtk
import gobject
import os
import pylab as P
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.backends.backend_gtkagg import \
  FigureCanvasGTKAgg as FigureCanvas
import matplotlib
ASPECT = matplotlib.__version__.find('0.87') > -1 and 'auto' or 'equal'

def iscomplex(a): return hasattr(a, "imag")

# Transforms for viewing different aspects of complex data
def ident_xform(data): return data
def abs_xform(data): return abs(data)
def phs_xform(data): return P.angle(data)
def real_xform(data): return data.real
def imag_xform(data): return data.imag

ui_info = \
'''<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <menuitem action='Save Image'/>
      <menuitem action='Save Montage'/>
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
      <menuitem action='Contour Plot'/>
    </menu>
  </menubar>
</ui>'''


##############################################################################
class sliceview (gtk.Window):
    "A Window class containing various plots and widgets"
    
    #mag_norm = normalize()
    #phs_norm = normalize(-pi, pi)
    _mouse_x = _mouse_y = None
    _dragging = False

    #-------------------------------------------------------------------------
    def __init__(self, data, dim_names=[], title="sliceview", cmap=P.cm.bone):
        self.data = P.asarray(data)

        # if data is complex, show the magnitude by default
        self.transform = iscomplex(data) and abs_xform or ident_xform

        # widget layout table
        table = gtk.Table(4, 2)        

        # control panel
        self.control_panel = \
          ControlPanel(data.shape, dim_names, iscomplex(data))
        self.control_panel.connect(
            self.spinnerHandler,
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
        self.conLevel = 1.0
        self.norm = None
        self.setNorm()

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
        #self.sliceplot.set_size_request(350,350)
        self.sliceplot.mpl_connect(
          'motion_notify_event', self.sliceMouseMotionHandler)
        self.sliceplot.mpl_connect(
          'button_press_event', self.sliceMouseDownHandler)
        self.sliceplot.mpl_connect(
          'button_release_event', self.sliceMouseUpHandler)
        def_scale = self.auto_scale_image()
        self.scrollwin.add_with_viewport(self.sliceplot)
        #table.attach(self.sliceplot, 1, 2, 2, 3)
        table.attach(self.scrollwin, 1, 2, 2, 3)

        # status
        self.status = StatusBar(self.sliceDataRange(), cmap)
        self.status.set_size_request(600,40)
        table.attach(self.status, 0, 2, 3, 4, xoptions=0, yoptions=0)

        # tool-bar
        merge = gtk.UIManager()
        merge.insert_action_group(self._create_action_group(def_scale), 0)

        try:
            mergeid = merge.add_ui_from_string(ui_info)
        except gobject.GError, msg:
            print "building menus failed: %s" % msg
        self.menubar = merge.get_widget("/MenuBar")

        table.attach(self.menubar, 0, 2, 0, 1, yoptions=0)

        self.updateDataRange()

        # initialize contour tools
        self.contour_tools = None
        
        # main window
        gtk.Window.__init__(self)
        self.connect("destroy", lambda x: gtk.main_quit())
        self.set_data("ui-manager", merge)
        self.add_accel_group(merge.get_accel_group())        
        # sum of widget height:
        # 27 for menu bar
        # 200 for row-plot, control panel
        # 400 for col-plot, scroll window
        # 40 for status bar
        # = 670
        self.set_default_size(600,670)
        self.set_title(title)
        self.set_border_width(3)
        self.add(table)
        self.show_all()
        P.show()

    #-------------------------------------------------------------------------
    def getRow(self):
        return self.getSlice()[self.control_panel.getRowIndex(),:]

    #-------------------------------------------------------------------------
    def getCol(self):
        return self.getSlice()[:,self.control_panel.getColIndex()]

    #-------------------------------------------------------------------------
    def getSlice(self):
        return self.transform(
          P.squeeze(self.data[self.control_panel.getIndexSlices()]))

    #-------------------------------------------------------------------------
    def updateRow(self):
        self.updateCrosshairs()
        self.rowplot.setData(self.getRow())

    #-------------------------------------------------------------------------
    def updateCol(self):
        self.updateCrosshairs()
        self.colplot.setData(self.getCol())

    #-------------------------------------------------------------------------
    def updateSlice(self):
        self.setNorm()
        cset = self.sliceplot.setData(self.getSlice(), norm=self.norm)
        self.rowplot.setData(self.getRow())
        self.colplot.setData(self.getCol())
        self.status.cbar.setRange(self.sliceDataRange(), norm=self.norm)
        if self.contour_tools is not None:
            self.contour_tools.draw_bar(cset)

    #-------------------------------------------------------------------------
    def sliceDataRange(self):
        flatSlice = P.ravel(self.getSlice())
        return P.amin(flatSlice), P.amax(flatSlice)

    #------------------------------------------------------------------------- 
    def updateDataRange(self):
        flat_data = self.transform(self.data.flat)
        data_min = P.amin(flat_data)
        data_max = P.amax(flat_data)
        self.rowplot.setDataRange(data_min, data_max)
        self.colplot.setDataRange(data_max, data_min)

    #-------------------------------------------------------------------------
    def spinnerHandler(self, adj):
        print "sliceview::spinnerHandler slice_dims", \
               self.control_panel.slice_dims

    #-------------------------------------------------------------------------
    def radioHandler(self, button, transform):
        if not button.get_active(): return
        self.transform = transform
        self.updateDataRange()
        self.updateSlice()

    #-------------------------------------------------------------------------
    def sliderHandler(self, adj):
        row_dim, col_dim= self.control_panel.slice_dims
        if adj.dim.index == row_dim: self.updateRow()
        elif adj.dim.index == col_dim: self.updateCol()
        else: self.updateSlice()

    #-------------------------------------------------------------------------
    def contrastHandler(self, adj):
        self.conLevel = self.control_panel.getContrastLevel()
        self.updateSlice()

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
    def updateCoords(self, y, x):

        # do nothing if coords haven't changed
        if x == self._mouse_x and y == self._mouse_y: return
        self._mouse_x, self._mouse_y = x, y

        # update statusbar element value label
        self.updateStatusLabel(y, x)

        # update crosshairs and projection plots if button down
        if self._dragging: self.updateProjections(y,x)

    #------------------------------------------------------------------------- 
    def updateStatusLabel(self, y, x):
        if x != None and y != None:
            text = "[%d,%d] = %.4f"%(y, x, self.getSlice()[y,x])
        else: text = ""
        self.status.setLabel(text)

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
    def setNorm(self):
        scale = -0.75*(self.conLevel-1.0) + 1.0
        dMin, dMax = self.sliceDataRange()

        # only scale the minimum value if it is below zero
        sdMin = dMin < 0 and dMin * scale or dMin

        # if the norm scalings haven't changed, don't change norm
        if self.norm and\
           (sdMin, dMax*scale) == (self.norm.vmin, self.norm.vmax): return

        # else set it to an appropriate scaling
        self.norm = self.transform == phs_xform and\
          P.normalize(-P.pi*scale, P.pi*scale) or P.normalize(sdMin, scale*dMax)
   
    #-------------------------------------------------------------------------

    def launch_contour_tool(self, action):
        self.contour_tools = ContourToolWin(self.sliceplot, self)

    def activate_action(self, action):
        self.dialog = gtk.MessageDialog(self, gtk.DIALOG_DESTROY_WITH_PARENT,
            gtk.MESSAGE_INFO, gtk.BUTTONS_CLOSE,
            'You activated action: "%s" of type "%s"' % (action.get_name(), type(action)))
        # Close dialog on user response
        self.dialog.connect ("response", lambda d, r: d.destroy())
        self.dialog.show()

    
    def ask_fname(self, prompt):
        dialog = gtk.FileChooserDialog(
            title=prompt,
            action=gtk.FILE_CHOOSER_ACTION_SAVE,
            parent=self,
            buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OK,gtk.RESPONSE_OK)
            )
        response = dialog.run()
        if response == gtk.RESPONSE_CANCEL:
            dialog.destroy()
            return
        fname = dialog.get_filename()
        dialog.destroy()
        fname = fname.rsplit(".")[-1] == "png" and fname or fname+".png"  
        return fname

    def save_png(self, action):
        # save a PNG of the current image and the current scaling
        fname = self.ask_fname("Save image as...")
        if fname is None:
            return
        im = self.sliceplot.getImage().make_image()
        im.write_png(fname)

    def save_png_montage(self, action):
        # make a montage PNG, for now make 5 slices to a row
        # make images 128x128 pix
        fname = self.ask_fname("Save montage as...")
        if fname is None:
            return
        #dshape = P.array(self.data.shape)
        nslice = self.data.shape[-3]
        cmap = self.sliceplot.getImage().cmap
        sdim = 128
        col_buf = 20
        row_buf = 50
        lr_buf = 20
        b_buf = 20
        #title_buf = 50
        ncol = 5 # hardwired for now
        nrow = int(nslice/ncol) + (nslice % ncol and 1 or 0)
        # get required height and width in pixels
        _ht = float(10 + nrow*(sdim + row_buf) + b_buf)
        _wd = float(2*lr_buf + ncol*sdim + (ncol-1)*col_buf)
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
                plane_slice[-3] = s
                Loff = (lr_buf + (col)*(sdim + col_buf))/_wd
                Boff = (b_buf + (nrow-row-1)*(sdim + row_buf))/_ht
                Xpct, Ypct = (sdim/_wd, sdim/_ht)
                ax = fig.add_axes([Loff, Boff, Xpct, Ypct])
                ax.imshow(P.squeeze(self.transform(self.data[plane_slice])),
                          cmap=cmap,
                          origin='lower',
                          interpolation='nearest')
                
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False)
                ax.set_frame_on(False)
                t = ax.set_title('Slice %d'%s)
                t.set_size(12)
        fig.savefig(fname, dpi=figdpi)

    def activate_radio_action(self, action, current):
        active = current.get_active()
        value = current.get_current_value()

        if active:
            dialog = gtk.MessageDialog(self, gtk.DIALOG_DESTROY_WITH_PARENT,
                gtk.MESSAGE_INFO, gtk.BUTTONS_CLOSE,
                "You activated radio action: \"%s\" of type \"%s\".\nCurrent value: %d" %
                (current.get_name(), type(current), value))

            # Close dialog on user response
            dialog.connect("response", lambda d, r: d.destroy())
            dialog.show()

    def cmap_handler(self, action, current):
        cmap = {
            0: P.cm.Blues,
            1: P.cm.Greens,
            2: P.cm.Greys,
            3: P.cm.Oranges,
            4: P.cm.Purples,
            5: P.cm.Reds,
            6: P.cm.Spectral,
            7: P.cm.autumn,
            8: P.cm.bone,
            9: P.cm.cool,
            10: P.cm.copper,
            11: P.cm.gist_earth,
            12: P.cm.gist_gray,
            13: P.cm.gist_heat,
            14: P.cm.gist_rainbow,
            15: P.cm.gray,
            16: P.cm.hot,
            17: P.cm.hsv,
            18: P.cm.jet,
            19: P.cm.spring,
            20: P.cm.summer,
            21: P.cm.winter,
            }[current.get_current_value()]
        self.sliceplot.setCmap(cmap)
        self.status.cbar.setCmap(cmap)

    def scale_handler(self, action, current):
        self.scale_image(current.get_current_value())

    def scale_image(self, scale):
        base_img_size = min(self.control_panel.getRowDim().size,
                            self.control_panel.getColDim().size)
        canvas_size = self.sliceplot.get_size_request()[0]
        canvas_size_real = self.sliceplot.get_width_height()[0]
        new_img_size = base_img_size*scale
        # If the new image requires a larger canvas, resize it.
        # Otherwise, make sure the canvas is at the default size
        if canvas_size < new_img_size+50:
            canvas_size_real = canvas_size = int(new_img_size + 50)
        elif canvas_size > 400 and new_img_size < 350:
            canvas_size = 350
            canvas_size_real = 396
        ax = self.sliceplot.getAxes()
        w = h = new_img_size/float(canvas_size_real)
        l = 15./canvas_size
        b = 1.0 - (new_img_size + 25.)/canvas_size_real
        ax.set_position([l,b,w,h])
        self.sliceplot.set_size_request(canvas_size,canvas_size)
        self.sliceplot.draw()

    def auto_scale_image(self):
        # try to find some scale that gets ~ 256x256 pixels
        base_img_size = min(self.control_panel.getRowDim().size,
                            self.control_panel.getColDim().size)
        P = round(256./base_img_size)
        new_img_size =  P*base_img_size
        canvas_size = 350
        canvas_size_real = 396
        ax = self.sliceplot.getAxes()
        w = h = new_img_size/float(canvas_size_real)
        l = 15./canvas_size
        b = 1.0 - (new_img_size + 25.)/canvas_size_real
        ax.set_position([l,b,w,h])
        self.sliceplot.set_size_request(canvas_size,canvas_size)
        #self.sliceplot.draw()
        return P
        
    def _create_action_group(self, default_scale):
        entries = (
            ( "FileMenu", None, "_File" ),
            ( "ToolsMenu", None, "_Tools" ),
            ( "SizeMenu", None, "_Image Size" ),
            ( "ColorMapping", None, "_Color Mapping"),
            ( "Save Image", gtk.STOCK_SAVE,
              "_Save Image", "<control>S",
              "Saves current slice as PNG",
              self.save_png ),
            ( "Save Montage", gtk.STOCK_SAVE,
              "_Save Montage", "<control><shift>S",
              "Saves all slices as a montage",
              self.save_png_montage ),
            ( "Quit", gtk.STOCK_QUIT,
              "_Quit", "<control>Q",
              "Quits",
              lambda action: self.destroy() ),
            ( "Contour Plot", None,
              "_Contour Plot", None,
              "Opens contour plot controls",
              self.launch_contour_tool )
        )

        size_toggles = (
            ( "1x", None, "_1x", None, "", 1 ),
            ( "2x", None, "_2x", None, "", 2 ),
            ( "4x", None, "_4x", None, "", 4 ),
            ( "6x", None, "_6x", None, "", 6 ),
            ( "8x", None, "_8x", None, "", 8 )
        )

        cmap_toggles = (
            ( "Blues", None, "_Blues", None, "", 0 ),
            ( "Greens", None, "_Greens", None, "", 1 ),
            ( "Greys", None, "_Greys", None, "", 2 ),
            ( "Oranges", None, "_Oranges", None, "", 3 ),
            ( "Purples", None, "_Purples", None, "", 4 ),
            ( "Reds", None, "_Reds", None, "", 5 ),
            ( "Spectral", None, "_Spectral", None, "", 6 ),
            ( "autumn", None, "_autumn", None, "", 7 ),
            ( "bone", None, "_bone", None, "", 8 ),
            ( "cool", None, "_cool", None, "", 9 ),
            ( "copper", None, "_copper", None, "", 10 ),
            ( "gist_earth", None, "_gist_earth", None, "", 11 ),
            ( "gist_gray", None, "_gist_gray", None, "", 12 ),
            ( "gist_heat", None, "_gist_heat", None, "", 13 ),
            ( "gist_rainbow", None, "_gist_rainbow", None, "", 14 ),
            ( "gray", None, "_gray", None, "", 15 ),
            ( "hot", None, "_hot", None, "", 16 ),
            ( "hsv", None, "_hsv", None, "", 17 ),
            ( "jet", None, "_jet", None, "", 18 ),
            ( "spring", None, "_spring", None, "", 19 ),
            ( "summer", None, "_summer", None, "", 20 ),
            ( "winter", None, "_winter", None, "", 21 ),
        )

        action_group = gtk.ActionGroup("WindowActions")
        action_group.add_actions(entries)
        action_group.add_radio_actions(size_toggles, int(default_scale),
                                       self.scale_handler)
        action_group.add_radio_actions(cmap_toggles, 8,
                                       self.cmap_handler)
        return action_group

##############################################################################
class ContourToolWin (gtk.Window):
    "A Window class defining a pop-up control widget"
    
    def __init__(self, obs_slice, parent):
        self.padre = parent
        self.sliceplot = obs_slice
        self.hbox = gtk.HBox(spacing=4)
        self.levSlider = gtk.VScale(gtk.Adjustment(7, 2, 20, 1, 1))
        self.levSlider.set_digits(0)
        self.levSlider.set_value_pos(gtk.POS_TOP)
        self.levSlider.get_adjustment().connect("value-changed",
                                                self.clevel_handler)
        self.hbox.pack_start(self.levSlider)
        self.fig = P.Figure(figsize=(1,4), dpi=80)
        self.cbar_ax = self.fig.add_axes([.1, .04, .55, .9])
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

    def _takedown(self, foo):
        self.sliceplot.killContour()
        self.padre.contour_tools = None
        foo.destroy()

    def setContours(self, levels):
        cset = self.sliceplot.doContours(levels)
        self.draw_bar(cset)

    def draw_bar(self, cset):
        self.cbar_ax.clear()
        self.fig.colorbar(cset, self.cbar_ax)
        self.figcanvas.draw()        

    def clevel_handler(self, adj):
        self.setContours(int(self.levSlider.get_value()))

##############################################################################
class Dimension (object):
    def __init__(self, index, size, name):
        self.index = index
        self.size = size
        self.name = name


##############################################################################
class DimSpinner (gtk.SpinButton):
    def __init__(self, name, value, start, end, handler):
        adj = gtk.Adjustment(0, start, end, 1, 1)
        adj.name = name
        gtk.SpinButton.__init__(self, adj, 0, 0)
        adj.connect("value-changed", handler)


##############################################################################
class DimSlider (gtk.HScale):
    def __init__(self, dim):
        adj = gtk.Adjustment(0, 0, dim.size-1, 1, 1)
        adj.dim = dim
        gtk.HScale.__init__(self, adj)
        self.set_digits(0)
        self.set_value_pos(gtk.POS_RIGHT)


##############################################################################
class ContrastSlider (gtk.HScale):
    def __init__(self):
        gtk.HScale.__init__(self, gtk.Adjustment(1.0, 0.05, 2.0, 0.05, 1))
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

        # spinner for row dimension
        #spinner_box = gtk.HBox()
        self.row_spinner = \
          DimSpinner("row", len(shape)-2, 0, len(shape)-2, self.spinnerHandler)
        #spinner_box.add(gtk.Label("Row:"))
        #spinner_box.add(self.row_spinner)

        # spinner for column dimension
        self.col_spinner = \
          DimSpinner("col", len(shape)-1, 1, len(shape)-1, self.spinnerHandler)
        #spinner_box.add(gtk.Label("Col:"))
        #spinner_box.add(self.col_spinner)
        #main_vbox.add(spinner_box)

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
        self.slice_dims = (self.dimensions[-2].index, self.dimensions[-1].index)

    #-------------------------------------------------------------------------
    def connect(self,
        spinner_handler, radio_handler, slider_handler, contrast_handler):
        "Connect control elements to the given handler functions."

        # connect slice orientation spinners
        self.row_spinner.get_adjustment().connect(
          "value-changed", spinner_handler)
        self.col_spinner.get_adjustment().connect(
          "value-changed", spinner_handler)

        # connect radio buttons
        for r in self.radios: r.connect("toggled", radio_handler, r.transform)

        # connect slice position sliders
        for s in self.sliders:
            s.get_adjustment().connect("value_changed", slider_handler)

        # connect contrast slider
        self.contrast_slider.get_adjustment().connect(
          "value_changed", contrast_handler)

    #-------------------------------------------------------------------------
    def getContrastLevel(self):
        return self.contrast_slider.get_adjustment().value

    #-------------------------------------------------------------------------
    def getDimPosition(self, dnum):
        return int(self.sliders[dnum].get_adjustment().value)

    #-------------------------------------------------------------------------
    def setDimPosition(self, dnum, index):
        return self.sliders[dnum].get_adjustment().set_value(int(index))

    #-------------------------------------------------------------------------
    def getRowIndex(self): return self.getDimPosition(self.slice_dims[0])

    #-------------------------------------------------------------------------
    def getColIndex(self): return self.getDimPosition(self.slice_dims[1])

    #------------------------------------------------------------------------- 
    def setRowIndex(self, index): self.setDimPosition(self.slice_dims[0],index)

    #------------------------------------------------------------------------- 
    def setColIndex(self, index): self.setDimPosition(self.slice_dims[1],index)

    #------------------------------------------------------------------------- 
    def getRowDim(self): return self.dimensions[self.slice_dims[0]]

    #------------------------------------------------------------------------- 
    def getColDim(self): return self.dimensions[self.slice_dims[1]]

    #-------------------------------------------------------------------------
    def getIndexSlices(self):
        return tuple([
          dim.index in self.slice_dims and\
            slice(0, dim.size) or\
            self.getDimPosition(dim.index)
          for dim in self.dimensions])

    #-------------------------------------------------------------------------
    def spinnerHandler(self, adj):
        newval = int(adj.value)
        row_adj = self.row_spinner.get_adjustment()
        col_adj = self.col_spinner.get_adjustment()

        if adj.name == "row" and newval >= int(col_adj.value):
            col_adj.set_value(newval+1)
        if adj.name == "col" and newval <= int(row_adj.value):
            row_adj.set_value(newval-1)

        self.slice_dims = (int(row_adj.value), int(col_adj.value))


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
        indices = P.arange(len(data))
        if not hasattr(self, "data"): ax.plot(indices, data)
        else: ax.lines[0].set_data(indices, data)
        ax.set_xlim(-.5, len(data)-.5)
        self.data = data
        self.draw()


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
        indices = P.arange(len(data))
        if not hasattr(self, "data"): ax.plot(data, indices)
        else: ax.lines[0].set_data(data, indices)
        ax.set_ylim(-.5,len(data)-.5)
        self.data = data
        self.draw()


##############################################################################
class SlicePlot (FigureCanvas):
    "A Canvas class containing a 2D matplotlib plot"    
    #-------------------------------------------------------------------------
    def __init__(self, data, x, y, cmap=P.cm.bone, norm=None):
        self.norm = None
        self.hasContours = False
        self.contourLevels = 7
        fig = P.Figure(figsize=P.figaspect(data), dpi=80)
        ax = fig.add_subplot(111)
        ax.yaxis.tick_right()
        ax.title.set_y(1.05) 
        FigureCanvas.__init__(self, fig)
        self.cmap = cmap
        self.setData(data, norm=norm)
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
    def getAxes(self): return self.figure.axes[0]

    #-------------------------------------------------------------------------
    def getImage(self):
        images = self.getAxes().images
        return len(images) > 0 and images[0] or None
        
    #-------------------------------------------------------------------------
    def setImage(self, image): self.getAxes().images[0] = image

    #-------------------------------------------------------------------------
    def setCmap(self, cmapObj):
        self.cmap = cmapObj
        self.setData(self.data, self.norm)
    #-------------------------------------------------------------------------
    def setData(self, data, norm=None):
        ax = self.getAxes()

        if self.getImage() is None:
            ax.imshow(data, interpolation="nearest",
              cmap=self.cmap, norm=self.norm, origin="lower")
        elif norm != self.norm:
            self.setImage(AxesImage(ax, interpolation="nearest",
              cmap=self.cmap, norm=norm, origin="lower"))
        
        self.getImage().set_data(data)
        self.getImage().set_cmap(self.cmap)
        self.norm = norm
        nrows, ncols = data.shape[:2]
        ax.set_xlim((0,ncols))
        ax.set_ylim((0,nrows))
        self.data = data
        if self.hasContours:
            return self.doContours(self.contourLevels)
        else:
            self.draw()
            return None

    def doContours(self, levels):
        self.hasContours = True
        self.contourLevels = levels
        ax = self.getAxes()
        ax.collections = []
        mn, mx = P.amin(self.data.flat), P.amax(self.data.flat)
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

    def killContour(self):
        ax = self.getAxes()
        ax.collections = []
        self.hasContours = False
        self.draw()
    
    #------------------------------------------------------------------------- 
    def setCrosshairs(self, x, y):
        row_data, col_data = self._crosshairs_data(x, y)
        row_line, col_line = self.crosshairs
        row_line.set_data(*row_data)
        col_line.set_data(*col_data)
        self.draw()

    #-------------------------------------------------------------------------
    def getEventCoords(self, event):
        if event.xdata is not None: x = int(event.xdata)
        else: x = None
        if event.ydata is not None:y = int(event.ydata)
        else: y = None
        if x < 0 or x >= self.data.shape[0]: x = None
        if y < 0 or y >= self.data.shape[1]: y = None
        return (y,x)


##############################################################################
class ColorBar (FigureCanvas):
    "A Canvas class showing the constrast scaling"
    #-------------------------------------------------------------------------
    def __init__(self, range, cmap=P.cm.bone, norm=None):
        fig = P.Figure(figsize = (5,0.5))
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
            r_pts = P.zeros((128,))
            tx = P.asarray([0])
        else:
            # make decently smooth gradient, try to include end-point
            delta = (dMax-dMin)/127
            r_pts = P.arange(dMin, dMax+delta, delta)
            # sometimes forcing the end-point breaks
            if len(r_pts) > 128: r_pts = P.arange(dMin, dMax, delta)

            # set up tick marks
            delta = (r_pts[-1] - r_pts[0])/7
            eps = 0.1 * delta
            tx = P.arange(r_pts[0], r_pts[-1], delta)
            # if the last tick point is very far away from the end,
            # add one more at the end
            if (r_pts[-1] - tx[-1]) > .75*delta:
                #there MUST be an easier way!
                a = tx.tolist()
                a.append(r_pts[-1])
                tx = P.asarray(a)
            # else if the last tick point is misleadingly close,
            # replace it with the true endpoint
            elif (r_pts[-1] - tx[-1]) > eps: tx[-1] = r_pts[-1]

        data = P.outerproduct(P.ones(5),r_pts)
        # need to clear axes because axis Intervals weren't updating
        ax.clear()
        ax.imshow(data, interpolation="nearest",
                  cmap=self.cmap, norm=norm, aspect=ASPECT,
                  extent=(r_pts[0], r_pts[-1], 0, 1))
        ax.images[0].set_data(data)
        ax.xaxis.set_ticks(tx)
        for tk in ax.xaxis.get_ticklabels(): tk.set_size(10.0)
        self.data = data
        self.draw()


##############################################################################
class StatusBar (gtk.Frame):
    
    #-------------------------------------------------------------------------
    def __init__(self, range, cmap):
        gtk.Frame.__init__(self)
        main_hbox = gtk.HBox()
        main_hbox.set_border_width(0)

        # neighborhood size selection (eg '5x5', '3x4')
        # these numbers refer to "radii", not box size
        self.entry = gtk.Entry(3)
        self.entry.set_size_request(40,25)

        # pixel value
        self.px_stat = gtk.Statusbar()
        self.px_stat.set_has_resize_grip(False)
        #self.px_stat.set_size_request(160,25)

        # neighborhood avg
        self.av_stat = gtk.Statusbar()
        self.av_stat.set_has_resize_grip(False)
        #self.av_stat.set_size_request(160,25)

        # try to label entry box
        #label = gtk.Label("Radius")
        #label.set_alignment(0, 0.2)
        #label.set_size_request(10,25)
        #label.set_line_wrap(True)

        # colorbar
        self.cbar = ColorBar(range, cmap=cmap)
        self.cbar.set_size_request(400,20)
        main_hbox.add(self.cbar)
 
        # pixel value
        self.label = gtk.Label()
        self.label.set_alignment(2, 0.5)
        self.label.set_size_request(140,20)
        self.label.set_line_wrap(True)
        main_hbox.add(self.label)
       
        self.px_context = self.px_stat.get_context_id("Pixel Value")
        self.av_context = self.av_stat.get_context_id("Neighborhood Avg")
        # default area to average
        self.entry.set_text('3x3')
        self.add(main_hbox)
        self.show_all()

    #-------------------------------------------------------------------------
    def report(self, event, data):
        if not (event.xdata and event.ydata):
            avbuf = pxbuf = "  clicked outside axes"
        else:
            y, x = int(event.ydata), int(event.xdata)
            pxbuf = "  pix val: %f"%data[y, x]
            avbuf = "  %ix%i avg: %s"%self.squareAvg(y, x, data)
        
        self.pop_items()
        self.push_items(pxbuf, avbuf)

    #-------------------------------------------------------------------------
    def squareAvg(self, y, x, data):
        areaStr = self.getText()
        #box is defined +/-yLim rows and +/-xLim cols
        #if for some reason areaStr was entered wrong, default to (1, 1)
        yLim, xLim = len(areaStr)==3 and\
                     (int(areaStr[0]), int(areaStr[2])) or (1, 1)
        if y < yLim or x < xLim or\
           y+yLim >= data.shape[0] or\
           x+xLim >= data.shape[1]:
            return (yLim, xLim, "outOfRange")

        indices = P.afromfunction(lambda yi,xi: y+yi-yLim + 1.0j*(x + xi-xLim),
                               (yLim*2+1, xLim*2+1))
        scale = indices.shape[0]*indices.shape[1]
        av = sum(map(lambda zi: data[int(zi.real), int(zi.imag)]/scale,
                     indices.flat))
        
        #return box dimensions and 7 significant digits of average
        return (yLim, xLim, str(av)[0:8])

    #-------------------------------------------------------------------------
    def getText(self): return self.entry.get_text()

    #-------------------------------------------------------------------------
    def setLabel(self, text):
        self.label.set_text(text)

    #-------------------------------------------------------------------------    
    def pop_items(self):
        self.av_stat.pop(self.av_context)
        self.px_stat.pop(self.px_context)

    #-------------------------------------------------------------------------
    def push_items(self, pxbuf, avbuf):
        self.av_stat.push(self.av_context, avbuf)
        self.px_stat.push(self.px_context, pxbuf)


##############################################################################
if __name__ == "__main__":
    from pylab import randn
    import pdb
    #pdb.run('sliceview(randn(6,6))', globals=globals(), locals=locals())
    pdb.run('sliceview(img.data)', globals=globals(), locals=locals())
    #sliceview(img.data.data)
