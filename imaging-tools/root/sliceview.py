#!/usr/bin/env python
import gtk
from pylab import Figure, figaspect, gci, show, amax, amin, squeeze, asarray,\
    cm, angle, normalize, pi, arange, ravel, ones, outerproduct, floor,\
    fromfunction, zeros
from matplotlib.image import AxesImage
from matplotlib.backends.backend_gtkagg import \
  FigureCanvasGTKAgg as FigureCanvas

def iscomplex(a): return hasattr(a, "imag")

# Transforms for viewing different aspects of complex data
def ident_xform(data): return data
def abs_xform(data): return abs(data)
def phs_xform(data): return angle(data)
def real_xform(data): return data.real
def imag_xform(data): return data.imag


##############################################################################
class DimSpinner (gtk.SpinButton):

    #-------------------------------------------------------------------------
    def __init__(self, name, value, start, end, handler):
        adj = gtk.Adjustment(0, start, end, 1, 1)
        adj.name = name
        gtk.SpinButton.__init__(self, adj, 0, 0)
        adj.connect("value-changed", handler)


##############################################################################
class DimSlider (gtk.HScale):

    #-------------------------------------------------------------------------
    def __init__(self, dim_num, dim_size, dim_name):
        adj = gtk.Adjustment(0, 0, dim_size-1, 1, 1)
        adj.dim_num = dim_num
        gtk.HScale.__init__(self, adj)
        self.set_digits(0)
        self.set_value_pos(gtk.POS_RIGHT)


##############################################################################
class ControlPanel (gtk.Frame):

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
        self.sliders = [DimSlider(*d) for d in self.dimensions]
        for slider, dimension in zip(self.sliders, self.dimensions):
            label = gtk.Label("%s:"%dimension[2])
            label.set_alignment(0, 0.5)
            main_vbox.pack_start(label, False, False, 0)
            main_vbox.pack_start(slider, False, False, 0)

        # slider for contrast adjustment
        label = gtk.Label("Contrast")
        label.set_alignment(0, 0.5)
        self.contrast_slider = gtk.HScale(
          gtk.Adjustment(1.0, 0.05, 2.0, 0.05, 1))
        self.contrast_slider.set_value_pos(gtk.POS_RIGHT)
        self.contrast_slider.set_digits(2)
        main_vbox.pack_start(label, False, False, 0)
        main_vbox.pack_start(self.contrast_slider, False, False, 0)

        self.add(main_vbox)

    #-------------------------------------------------------------------------
    def _init_dimensions(self, dim_sizes, dim_names):
        self.dimensions = []
        num_dims = len(dim_sizes)
        num_names = len(dim_names)
        if num_names != num_dims:
            dim_names = ["Dim %s"%i for i in range(num_dims)]
        for dim_num, (dim_size, dim_name) in\
          enumerate(zip(dim_sizes, dim_names)):
            self.dimensions.append( (dim_num, dim_size, dim_name) )
        self.slice_dims = (self.dimensions[-2][0], self.dimensions[-1][0])

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
    def getDimIndex(self, dnum):
        return int(self.sliders[dnum].get_adjustment().value)

    #-------------------------------------------------------------------------
    def getRowIndex(self): return self.getDimIndex(self.slice_dims[0])

    #-------------------------------------------------------------------------
    def getColIndex(self): return self.getDimIndex(self.slice_dims[1])

    #-------------------------------------------------------------------------
    def getSlices(self):
        return tuple([
          dnum in self.slice_dims and slice(0, dsize) or self.getDimIndex(dnum)
          for dnum, dsize, _ in self.dimensions])

    #-------------------------------------------------------------------------
    def spinnerHandler(self, adj):
        newval = int(adj.value)
        row_adj = self.row_spinner.get_adjustment()
        col_adj = self.col_spinner.get_adjustment()

        if adj.name == "row":
            if newval >= int(col_adj.value):
                col_adj.set_value(newval+1)
        if adj.name == "col":
            if newval <= int(row_adj.value):
                row_adj.set_value(newval-1)

        self.slice_dims = (int(row_adj.value), int(col_adj.value))


##############################################################################
class RowPlot (FigureCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, data):
        fig = Figure(figsize=(3., 6.))
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
        indices = range(len(data))
        if not hasattr(self, "data"): ax.plot(indices, data)
        else: ax.lines[0].set_data(indices, data)
        ax.set_xlim(-.5, len(data)-.5)
        self.data = data
        self.draw()


##############################################################################
class ColPlot (FigureCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, data):
        fig = Figure(figsize=(6., 3.))
        fig.add_axes([0.1, 0.1, 0.85, 0.85])
        FigureCanvas.__init__(self, fig)
        self.setData(data)

    #-------------------------------------------------------------------------
    def setDataRange(self, data_min, data_max):
        self.figure.axes[0].set_xlim(data_min, data_max)

    #-------------------------------------------------------------------------
    def setData(self, data):
        ax = self.figure.axes[0]
        indices = range(len(data))
        if not hasattr(self, "data"): ax.plot(data, indices)
        else: ax.lines[0].set_data(data, indices)
        ax.set_ylim(len(data)-.5, -.5)
        self.data = data
        self.draw()


##############################################################################
class SlicePlot (FigureCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, data, cmap=cm.bone, norm=None):
        self.norm = None
        fig = Figure(figsize=figaspect(data))
        ax  = fig.add_axes([0.05, 0.1, 0.85, 0.85])
        ax.yaxis.tick_right()
        ax.title.set_y(1.05) 
        FigureCanvas.__init__(self, fig)
        self.cmap = cmap
        self.setData(data, norm=norm)

    #-------------------------------------------------------------------------
    def getAxes(self): return self.figure.axes[0]

    #-------------------------------------------------------------------------
    def getImage(self):
        images = self.getAxes().images
        return len(images) > 0 and images[0] or None
        
    #-------------------------------------------------------------------------
    def setImage(self, image): self.getAxes().images[0] = image

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
        self.norm = norm
        nrows, ncols = data.shape[:2]
        ax.set_xlim((0,ncols))
        ax.set_ylim((nrows,0))
        self.data = data
        self.draw()


##############################################################################
class ColorBar (FigureCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, range, cmap=cm.bone, norm=None):
        fig = Figure(figsize = (5,0.5))
        fig.add_axes((0.05, 0.55, 0.9, 0.3))
        FigureCanvas.__init__(self, fig)
        self.figure.axes[0].yaxis.set_visible(False)
        self.cmap = cmap
        self.draw()
        self.setRange(range, norm=norm)

    #-------------------------------------------------------------------------
    def setRange(self, range, norm=None):
        self.norm = norm
        dMin, dMax = range
        ax = self.figure.axes[0]

        if dMin == dMax:
            r_pts = zeros((128,))
            tx = asarray([0])
        else:
            # make decently smooth gradient, try to include end-point
            delta = (dMax-dMin)/127
            r_pts = arange(dMin, dMax+delta, delta)
            # sometimes forcing the end-point breaks
            if len(r_pts) > 128: r_pts = arange(dMin, dMax, delta)

            # set up tick marks
            delta = (r_pts[-1] - r_pts[0])/7
            eps = 0.1 * delta
            tx = arange(r_pts[0], r_pts[-1], delta)
            # if the last tick point is very far away from the end,
            # add one more at the end
            if (r_pts[-1] - tx[-1]) > .75*delta:
                #there MUST be an easier way!
                a = tx.tolist()
                a.append(r_pts[-1])
                tx = asarray(a)
            # else if the last tick point is misleadingly close,
            # replace it with the true endpoint
            elif (r_pts[-1] - tx[-1]) > eps: tx[-1] = r_pts[-1]

        data = outerproduct(ones(5),r_pts)
        # need to clear axes because axis Intervals weren't updating
        ax.clear()
        ax.imshow(data, interpolation="nearest",
              cmap=self.cmap, norm=norm, extent=(r_pts[0], r_pts[-1], 0, 1))
        ax.images[0].set_data(data)
        ax.xaxis.set_ticks(tx)
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
                         (int(areaStr[0]), int(areaStr[2])) or\
                         (1, 1)
        if y < yLim or x < xLim or\
           y+yLim >= data.shape[0] or\
           x+xLim >= data.shape[1]:
            return (yLim, xLim, "outOfRange")

        indices = fromfunction(lambda yi,xi: y+yi-yLim + 1.0j*(x + xi-xLim),
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
class sliceview (gtk.Window):
    #mag_norm = normalize()
    #phs_norm = normalize(-pi, pi)
    _mouse_x = _mouse_y = None

    #-------------------------------------------------------------------------
    def __init__(self, data, dim_names=[], title="sliceview", cmap=cm.bone):
        self.data = asarray(data)

        # if data is complex, show the magnitude by default
        self.transform = iscomplex(data) and abs_xform or ident_xform

        # widget layout table
        table = gtk.Table(3, 2)

        # control panel
        self.control_panel = \
          ControlPanel(data.shape, dim_names, iscomplex(data))
        self.control_panel.connect(
            self.spinnerHandler,
            self.radioHandler,
            self.sliderHandler,
            self.contrastHandler)
        self.control_panel.set_size_request(200, 200)
        table.attach(self.control_panel, 0, 1, 0, 1)

        # row plot
        self.rowplot = RowPlot(self.getRow())
        self.rowplot.set_size_request(400, 200)
        table.attach(self.rowplot, 1, 2, 0, 1)

        # column plot
        self.colplot = ColPlot(self.getCol())
        self.colplot.set_size_request(200, 400)
        table.attach(self.colplot, 0, 1, 1, 2)
        
        # Set up normalization BEFORE plotting images.
        # Contrast level of 1.0 gives default normalization (changed by
        # contrast slider).
        self.conLevel = 1.0
        self.norm = None
        self.setNorm()

        # slice image
        self.sliceplot = SlicePlot(self.getSlice(), cmap=cmap, norm=self.norm)
        self.sliceplot.mpl_connect('button_press_event', self.sliceClickHandler)
        self.sliceplot.mpl_connect('motion_notify_event', self.sliceMouseMotionHandler)
        self.sliceplot.set_size_request(400, 400)
        table.attach(self.sliceplot, 1, 2, 1, 2)

        # status
        self.status = StatusBar(self.sliceDataRange(), cmap)
        self.status.set_size_request(200,30)
        table.attach(self.status, 0, 2, 2, 3)

        self.updateDataRange()

        # main window
        gtk.Window.__init__(self)
        self.connect("destroy", lambda x: gtk.main_quit())
        self.set_default_size(400,300)
        self.set_title(title)
        self.set_border_width(3)
        self.add(table)
        self.show_all()
        show()

    #-------------------------------------------------------------------------
    def getRow(self):
        return self.getSlice()[self.control_panel.getRowIndex(),:]

    #-------------------------------------------------------------------------
    def getCol(self):
        return self.getSlice()[:,self.control_panel.getColIndex()]

    #-------------------------------------------------------------------------
    def getSlice(self):
        return self.transform(
          squeeze(self.data[self.control_panel.getSlices()]))

    #-------------------------------------------------------------------------
    def updateRow(self): self.rowplot.setData(self.getRow())

    #-------------------------------------------------------------------------
    def updateCol(self): self.colplot.setData(self.getCol())

    #-------------------------------------------------------------------------
    def updateSlice(self):
        self.setNorm()
        self.sliceplot.setData(self.getSlice(), norm=self.norm)
        self.rowplot.setData(self.getRow())
        self.colplot.setData(self.getCol())
        self.status.cbar.setRange(self.sliceDataRange(), norm=self.norm)

    #-------------------------------------------------------------------------
    def sliceDataRange(self):
        flatSlice = ravel(self.getSlice())
        return amin(flatSlice), amax(flatSlice)

    #------------------------------------------------------------------------- 
    def updateDataRange(self):
        flat_data = self.transform(self.data.flat)
        data_min = amin(flat_data)
        data_max = amax(flat_data)
        self.rowplot.setDataRange(data_min, data_max)
        self.colplot.setDataRange(data_max, data_min)

    #-------------------------------------------------------------------------
    def spinnerHandler(self, adj):
        print "VolumeViewer::spinnerHandler slice_dims", \
               self.control_panel.slice_dims

    #-------------------------------------------------------------------------
    def radioHandler(self, button, transform):
        if not button.get_active(): return
        self.transform = transform
        self.updateDataRange()
        self.updateSlice()

    #-------------------------------------------------------------------------
    def sliderHandler(self, adj):
        row_dim_num, col_dim_num = self.control_panel.slice_dims
        if adj.dim_num == row_dim_num: self.updateRow()
        elif adj.dim_num == col_dim_num: self.updateCol()
        else: self.updateSlice()

    #-------------------------------------------------------------------------
    def contrastHandler(self, adj):
        self.conLevel = self.control_panel.getContrastLevel()
        self.updateSlice()

    #-------------------------------------------------------------------------
    def sliceClickHandler(self, event):
        self.status.report(event, self.getSlice())

    #-------------------------------------------------------------------------
    def sliceMouseMotionHandler(self, event):
        if event.xdata is None or event.ydata is None: x = y = None
        else: x, y = int(event.xdata), int(event.ydata)
        if x == self._mouse_x and y == self._mouse_y: return
        self._mouse_x, self._mouse_y = x, y
        slice = self.getSlice()
        width, height = slice.shape[0], slice.shape[1]

        text = ""
        if x != None and y != None and\
           x >= 0 and x < width and\
           y >= 0 and y < height:
            text = "[%d,%d] = %.4f"%(y, x, slice[y,x])
        self.status.setLabel(text)
        
    #------------------------------------------------------------------------- 
    def setNorm(self):
        scale = -0.75*(self.conLevel-1.0) + 1.0
        dMin, dMax = self.sliceDataRange()

        # only scale the minimum value if it is below zero (?)
        sdMin = dMin < 0 and dMin * scale or dMin

        # if the norm scalings haven't changed, don't change norm
        if self.norm and\
           (sdMin, dMin*scale) == (self.norm.vmin, self.norm.vmax): return

        # else set it to an appropriate scaling
        self.norm = self.transform == phs_xform and\
          normalize(-pi*scale, pi*scale) or normalize(sdMin, scale*dMax)
   

##############################################################################
if __name__ == "__main__":
    from pylab import randn
    sliceview(randn(6,6))
