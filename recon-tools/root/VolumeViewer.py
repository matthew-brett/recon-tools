#!/usr/bin/env python
import gtk
from pylab import Figure, figaspect, gci, show, amax, amin, squeeze, asarray, cm
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas


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
    def __init__(self, shape, dim_names=[]):
        self._init_dimensions(shape, dim_names)
        gtk.Frame.__init__(self)
        main_vbox = gtk.VBox()

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

        # slider for each data dimension
        self.sliders = [DimSlider(*d) for d in self.dimensions]
        for slider, dimension in zip(self.sliders, self.dimensions):
            label = gtk.Label("%s:"%dimension[2])
            label.set_alignment(0, 0.5)
            main_vbox.pack_start(label, False, False, 0)
            main_vbox.pack_start(slider, False, False, 0)

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
    def connect(self, spinner_handler, slider_handler):
        self.row_spinner.get_adjustment().connect(
          "value-changed", spinner_handler)
        self.col_spinner.get_adjustment().connect(
          "value-changed", spinner_handler)
        for s in self.sliders:
            s.get_adjustment().connect("value_changed", slider_handler)

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
    def __init__(self, data, cmap=cm.bone):
        fig = Figure(figsize=figaspect(data))
        ax  = fig.add_axes([0.05, 0.1, 0.85, 0.85])
        ax.yaxis.tick_right()
        ax.title.set_y(1.05) 
        FigureCanvas.__init__(self, fig)
        self.cmap = cmap
        self.setData(data)

    #-------------------------------------------------------------------------
    def setData(self, data):
        ax = self.figure.axes[0]
        if not hasattr(self, "data"):
            ax.imshow(data, interpolation="nearest", cmap=self.cmap)
        else: ax.images[0].set_data(data)
        nr, nc = data.shape[:2]
        ax.set_xlim((0,nc))
        ax.set_ylim((nr,0))
        self.data = data
        self.draw()


##############################################################################
class VolumeViewer (gtk.Window):

    #-------------------------------------------------------------------------
    def __init__(self, data, dim_names=[], title="VolumeViewer", cmap=cm.bone):
        self.adjustments = []
        self.data = asarray(data)

        # widget layout table
        table = gtk.Table(2, 2)

        # control panel
        self.control_panel = ControlPanel(data.shape, dim_names)
        self.control_panel.connect(self.spinnerHandler, self.sliderHandler)
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

        # slice image
        self.sliceplot = SlicePlot(self.getSlice(), cmap=cmap)
        self.sliceplot.set_size_request(400, 400)
        table.attach(self.sliceplot, 1, 2, 1, 2)

        self.updateDataRange()

        # main window
        gtk.Window.__init__(self)
        self.connect("destroy", lambda x: gtk.main_quit())
        self.set_default_size(400,300)
        self.set_title(title)
        self.set_border_width(5)
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
        return squeeze(self.data[self.control_panel.getSlices()])

    #-------------------------------------------------------------------------
    def updateRow(self): self.rowplot.setData(self.getRow())

    #-------------------------------------------------------------------------
    def updateCol(self): self.colplot.setData(self.getCol())

    #-------------------------------------------------------------------------
    def updateSlice(self):
        self.sliceplot.setData(self.getSlice())
        self.rowplot.setData(self.getRow())
        self.colplot.setData(self.getCol())

    #-------------------------------------------------------------------------
    def updateDataRange(self):
        flat_data = self.data.flat
        data_min = amin(flat_data)
        data_max = amax(flat_data)
        self.rowplot.setDataRange(data_min, data_max)
        self.colplot.setDataRange(data_max, data_min)

    #-------------------------------------------------------------------------
    def spinnerHandler(self, adj):
        print "VolumeViewer::spinnerHandler slice_dims", self.control_panel.slice_dims

    #-------------------------------------------------------------------------
    def sliderHandler(self, adj):
        row_dim_num, col_dim_num = self.control_panel.slice_dims
        if adj.dim_num == row_dim_num: self.updateRow()
        elif adj.dim_num == col_dim_num: self.updateCol()
        else: self.updateSlice()


##############################################################################
if __name__ == "__main__":
    from pylab import randn
    viewer = VolumeViewer(randn(6,6))
