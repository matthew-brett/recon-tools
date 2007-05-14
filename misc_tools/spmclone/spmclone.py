#!/usr/bin/env python
import gtk
import gobject
import os
import sys
import pylab as P
import numpy as N
from matplotlib.nxutils import points_inside_poly
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import Widget
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.backends.backend_gtkagg import \
     FigureCanvasGTKAgg as FigureCanvas
import matplotlib

from recon.imageio import readImage
from recon import util
from odict import odict
from vertex_tools import get_edge_polys

ui_info = \
'''<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <menuitem action='Quit'/>
    </menu>
    <menu action='ToolsMenu'>
      <menuitem action='VOI'/>
    </menu>
  </menubar>
</ui>'''

AX, COR, SAG = range(3)
interp_types = ['nearest', 'bilinear', 'sinc']
interp_lookup = odict([(num,name) for num,name in enumerate(interp_types)])

class spmclone (gtk.Window):
    
    def __init__(self, image):
        table = gtk.Table(4, 2)

        self.image = image
        self.r0 = N.array([self.image.z0, self.image.y0, self.image.x0])
        self.dr = N.array([self.image.zsize, self.image.ysize,
                           self.image.xsize])
        self.dimlengths = self.dr * N.array(image.shape)
        # get vox origin from zyx origin (can't use transform because
        # (these zyx are in the image's native space)
        [z0,y0,x0] = N.round(N.array([image.z0,image.y0,image.x0])/self.dr).tolist()
        self.vox_coords = [z0,y0,x0]        
        # orientation xform maps vx-space:xyz-space (Right,Anterior,Superior)
        # numpy arrays are in C-ordering, so I'm going to reverse the
        # rows and columns of the xform
        # Also, these xforms are right-handed, but ANALYZE parameter data
        # are left handed, so I'm going to reverse the sign of the row
        # mapping to x
        # (makes as if the images are right handed??)
        self.xform = image.orientation_xform.tomatrix()
        self.xform = util.reverse(util.reverse(self.xform,axis=-1),axis=-2)
        self.xform[-1] = abs(self.xform[-1])
        self.zoom = 0

        # Make the ortho plots ---

        # I'm transforming [AX,COR,SAG] so that this list informs each
        # sliceplot what dimension it slices in the image array
        # (eg for coronal data, the coronal plot slices the 0th dim)
        [globals()['AX'],
         globals()['COR'],
         globals()['SAG']] = N.dot(self.xform, N.arange(3)).astype(N.int32)
        self.setNorm()
        self.slice_patches = None
        self.ax_plot=SlicePlot(self.data_xform(AX), 0, 0, AX,
                               norm=self.norm,
                               extent=[-self.dimlengths[SAG]/2.,
                                       self.dimlengths[SAG]/2.,
                                       -self.dimlengths[COR]/2.,
                                       self.dimlengths[COR]/2.])
        
        self.cor_plot=SlicePlot(self.data_xform(COR), 0, 0, COR,
                                norm=self.norm,
                                extent=[-self.dimlengths[SAG]/2.,
                                        self.dimlengths[SAG]/2.,
                                        -self.dimlengths[AX]/2.,
                                        self.dimlengths[AX]/2.])
        
        self.sag_plot=SlicePlot(self.data_xform(SAG), 0, 0, SAG,
                                norm=self.norm,
                                extent=[-self.dimlengths[COR]/2.,
                                        self.dimlengths[COR]/2.,
                                        -self.dimlengths[AX]/2.,
                                        self.dimlengths[AX]/2.])
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
        self.statusbox = DisplayStatus(tuple(self.vox_coords),
                                       tuple(self.zyx_coords()))
        table.attach(self.statusbox, 0, 1, 3, 4)
        self.statusbox.set_size_request(300,300)
        #table.set_row_spacing(1,25)
        # heights = 800
        # 250 plot 1
        # 250 plot 2
        # 370 info stuff
        # 30 menubar
        self.connect_crosshair_id = []
        self.connectCrosshairEvents()
        
        gtk.Window.__init__(self)
        self.connect("destroy", lambda x: gtk.main_quit())
        self.set_data("ui-manager", merge)
        self.add_accel_group(merge.get_accel_group())
        self.set_default_size(600,730)
        self.set_border_width(3)
        self.add(table)
        self.show_all()
        self.setUpAxesSize()        
        P.show()
    #-------------------------------------------------------------------------
    def zyx_coords(self):
        "makes the transformation from current vox to zyx coords"
        return N.dot(self.xform, self.vox_coords*self.dr - self.r0).tolist()
    
    #-------------------------------------------------------------------------
    def zyx2vox(self, zyx):
        "makes the transformation from zyx coords to a voxel position"
        zyx = N.asarray(zyx)
        xform = N.linalg.inv(self.xform)
        r_img = N.dot(xform, zyx)
        return N.round( (r_img + self.r0)/self.dr ).tolist()

    #-------------------------------------------------------------------------
    def setNorm(self):
        "sets the whitepoint and blackpoint"
        ordered_image = N.sort(self.image[:].flatten())
        Npt = ordered_image.shape[0]
        p01 = ordered_image[N.round(Npt*.01)]
        p99 = ordered_image[N.round(Npt*.99)]
        self.norm = P.normalize(vmin = p01, vmax = p99)

    #-------------------------------------------------------------------------
    def data_xform(self, slice_idx):
        """
        Given the a sliceplots slicing index and the current vox position,
        get a slice of data.
        """
        slicer = [slice(0,d) for d in self.image.shape]
        slicer[slice_idx] = self.vox_coords[slice_idx]
        slicer = tuple(slicer)
        if self.is_xpose(slice_idx):
            return N.swapaxes(self.image[slicer], 0, 1)
        else:
            return self.image[slicer]

    #-------------------------------------------------------------------------
    def is_xpose(self, slice_idx):
        """
        Based on the vox->zyx mapping, see if the data that is sliced in
        this direction should be transposed. This can be found by examining
        the submatrix that maps to the 2 dimensions of the slice.
        """
        M = self.xform.copy()
        row_idx = range(3)
        col_idx = range(3)
        col = slice_idx
        row = (abs(M[:,col]) == 1).nonzero()[0]
        row_idx.remove(row)
        col_idx.remove(col)
        # do some numpy "fancy" slicing to find the sub-matrix
        # if it's not an identity, then transpose the data slice
        Msub = M[row_idx][:,col_idx]
        # if this is not an identity, then the axes must be unswapped
        return not Msub[0,0]

    #-------------------------------------------------------------------------
    def updateSlices(self, sliceplots=None):
        if not sliceplots:
            sliceplots = self.sliceplots
        for sliceplot in sliceplots:
            idx = sliceplot.slice_idx
            sliceplot.setData(self.data_xform(idx), self.norm)
            if self.slice_patches is not None:
                p_idx = int(self.vox_coords[idx])
                sliceplot.showPatches(self.slice_patches[idx][p_idx])

    #-------------------------------------------------------------------------
    def updateCrosshairs(self):
        for s,sliceplot in enumerate(self.sliceplots):
            idx = sliceplot.slice_idx
            zyx = self.zyx_coords()
            zyx.pop(s)
            ud,lr = zyx
            sliceplot.setCrosshairs(lr,ud)

    #-------------------------------------------------------------------------
    def setUpAxesSize(self):
        "Scale the axes appropriately for the image dimensions"
        # assume that image resolution is isotropic in dim2 and dim1
        # (not necessarily in dim0)
        # want the isotropic resolution plot to be 215x215 pixels
        xy_imgsize = 215.
        ref_size = self.dimlengths[-1]
        slicing = [AX,COR,SAG]
        for sliceplot in self.sliceplots:
            dims_copy = N.take(self.dimlengths, slicing).tolist()
            ax = sliceplot.getAxes()
            s_idx = sliceplot.slice_idx
            dims_copy.remove(self.dimlengths[s_idx])
            slice_y, slice_x = dims_copy
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
        trans_sliceplots = {
            self.sliceplots[AX]: (self.sliceplots[SAG], self.sliceplots[COR]),
            self.sliceplots[COR]: (self.sliceplots[SAG], self.sliceplots[AX]),
            self.sliceplots[SAG]: (self.sliceplots[COR], self.sliceplots[AX]),
            }.get(sliceplot)
        trans_idx = (trans_sliceplots[0].slice_idx,
                     trans_sliceplots[1].slice_idx)
        # where do left-right and up-down cut across in zyx space?
        trans_ax = {
            AX: (2, 1), # x,y
            COR: (2, 0), # x,z
            SAG: (1, 0), # y,z
        }.get(sliceplot.slice_idx)
        zyx_clicked = self.zyx_coords()
        zyx_clicked[trans_ax[0]] = lr
        zyx_clicked[trans_ax[1]] = ud
        vox = self.zyx2vox(zyx_clicked)
        self.vox_coords[trans_idx[0]] = vox[trans_idx[0]]
        self.vox_coords[trans_idx[1]] = vox[trans_idx[1]]
        self.updateSlices(sliceplots=trans_sliceplots)
        self.updateCrosshairs()
        # make text to update the statusbox label's
        self.statusbox.set_vox_text(self.vox_coords)
        self.statusbox.set_zyx_text(self.zyx_coords())

    #-------------------------------------------------------------------------
    def VOI_handler(self, action):
        # turns off the crosshairs and sets up the VOI drawing sequence
        self.connectCrosshairEvents(mode="disable")
        for sliceplot in self.sliceplots:
            sliceplot.getAxes().patches = []
        ax_plot = self.sliceplots[0]
        self.mask = N.ones(self.image.shape, N.int32)
        #self.lasso_id = ax_plot.mpl_connect("button_press_event",
        #                                    self.lasso_handler)
        self.lasso_plot = ax_plot
        self.lasso_id = ax_plot.mpl_connect("button_press_event",
                                            self.new_lasso_handler)
        ax_plot.draw_idle()        

    #-------------------------------------------------------------------------
##     def lasso_handler(self, event):
##         if self.sliceplots[0].widgetlock.locked(): return
##         if event.inaxes is None or \
##            event.inaxes is not self.sliceplots[0].getAxes(): return
##         self.lasso = MyLasso(event.inaxes, (event.xdata, event.ydata),
##                              self.ax_mask_from_lasso)
##         self.sliceplots[0].widgetlock(self.lasso)

    #-------------------------------------------------------------------------
    def new_lasso_handler(self, event):
        plot = self.lasso_plot
        if plot.widgetlock.locked(): return
        if event.inaxes is None or \
           event.inaxes is not plot.getAxes(): return
        # disable the callback here, because a new button_press_event
        # will be added by the polydraw
        plot.mpl_disconnect(self.lasso_id)        
        plot.getAxes().patches = []
        # This method gets called when the polygon-drawing is done--
        # it receives a list of vertices, from which it updates the mask,
        # and then puts the drawing into the next stage
        def mask_from_lasso(verts):
            slicing = [AX,COR,SAG]
            slicing.remove(plot.slice_idx)
            shape = self.data_xform(plot.slice_idx).shape
            sizes = N.take(self.dr, slicing)
            offsets = N.take(self.r0, slicing)
            rx = N.arange(shape[-1])*sizes[-1] - offsets[-1]
            ry = N.arange(shape[-2])*sizes[-2] - offsets[-2]
            #print rx
            #print ry
            x,y = P.meshgrid(rx,ry)
            xys = zip(x.flatten(), y.flatten())
            inside = points_inside_poly(xys, verts)
            mask = N.reshape(inside, shape)
            if self.is_xpose(plot.slice_idx):
                mask = N.swapaxes(mask, 0, 1)
            slices = [slice(0,d) for d in mask.shape]
            slices.insert(plot.slice_idx, None)
            #print shape, mask.shape, self.mask.shape, plot.slice_idx
            self.mask[:] = self.mask[:] * mask[tuple(slices)]
            #print inside.sum()
            poly = PolyCollection([verts,], facecolors=(0.0,0.8,0.2,0.4))
            plot.getAxes().add_patch(poly)
            plot.draw_idle()
            #plot.mpl_disconnect(self.lasso_id)
            plot.widgetlock.release(self.lasso)
            if plot.slice_idx is AX:
                # draw COR and SAG rectangles, set off lasso handler on COR
                (cor_plot, sag_plot) = self.sliceplots[1:]
                (cor_ax, sag_ax) = (cor_plot.getAxes(), sag_plot.getAxes())

                height = self.data_xform(SAG).shape[-2]*self.dr[AX]
                y0 = -self.r0[AX]
                proj_axes = self.is_xpose(AX) and (-1,-2) or (-2,-1)
                lr_proj = mask.sum(axis=proj_axes[0]).nonzero()[0]
                #print lr_proj
                cor_width = rx[lr_proj[-1]] - rx[lr_proj[0]]
                cor_xy = (rx[lr_proj[0]], y0) 

                ud_proj = mask.sum(axis=proj_axes[1]).nonzero()[0]
                sag_width = ry[ud_proj[-1]] - ry[ud_proj[0]]
                sag_xy = (ry[ud_proj[0]], y0)

                props = dict(alpha=0.4, facecolor=(0.0,0.8,0.2))
                
                cor_rect_trans = P.blend_xy_sep_transform(cor_ax.transData,
                                                         cor_ax.transAxes)
                
                cor_rect = Rectangle(cor_xy, cor_width, height,
                                     visible=True, **props)
                cor_ax.add_patch(cor_rect)
                
                sag_rect = Rectangle(sag_xy, sag_width, height,
                                    visible=True, **props)
                sag_ax.add_patch(sag_rect)
                cor_plot.draw()
                sag_plot.draw()
                self.lasso_plot = cor_plot
                self.lasso_id = cor_plot.mpl_connect("button_press_event",
                                                     self.new_lasso_handler)
                #open("mask.dat", "wb").write(mask.tostring())
                
            elif plot.slice_idx is COR:
                # draw SAG rectangle, set off lasso handler on SAG
                sag_plot = self.sliceplots[-1]
                sag_ax = sag_plot.getAxes()
                proj_ax = self.is_xpose(COR) and -2 or -1
                ud_proj = mask.sum(axis=proj_ax).nonzero()[0]
                #print ud_proj
                #print "len(rx)=",len(rx),"len(ry)=",len(ry)
                ud_height = ry[ud_proj[-1]] - ry[ud_proj[0]]

                old_rect = sag_ax.patches[0]
                #old_rect.set_transform(sag_ax.get_transform())
                old_rect.set_y(ry[ud_proj[0]])

                old_rect.set_height(ud_height)

                sag_ax.draw_artist(old_rect)
                sag_plot.draw()
                self.lasso_id = sag_plot.mpl_connect("button_press_event",
                                                     self.new_lasso_handler)
                self.lasso_plot = sag_plot
                
            else:
                # clean up and reactivate regular callbacks
                self.lasso_plot = None
                self.connectCrosshairEvents()
                #open("mask.dat", "wb").write(self.mask.tostring())
                self.build_patches()
                print "patches built"

        self.lasso = MyPolyDraw(event.inaxes, (event.xdata, event.ydata),
                                mask_from_lasso)
        plot.widgetlock(self.lasso)
        
    #-------------------------------------------------------------------------
    def build_patches(self):
        """
        When defining the mask is complete, build the patches (polygons) for
        every slice in the 3 directions (a little time consuming).
        """
        self.slice_patches = [[],[],[]]
        slice_idxs = [s.slice_idx for s in self.sliceplots]
        for idx in slice_idxs:
            slicing = [AX,COR,SAG]
            slicing.remove(idx)
            slicer = [slice(0,d) for d in self.mask.shape]
            for d in xrange(self.mask.shape[idx]):
                slicer[idx] = d
                # get_edge_polys takes a mask slice and finds an ordered
                # path around the edge of each unmasked region, returning
                # a list of sorted vertex lists
                if self.is_xpose(idx):
                    polys = get_edge_polys(N.swapaxes(self.mask[tuple(slicer)],0,1))
                else:
                    polys = get_edge_polys(self.mask[tuple(slicer)])
                polygons = []
                # need to convert indices to x,y points
                for p in range(len(polys)):
                    xx = N.array([x for x,y in polys[p]])
                    yy = N.array([y for x,y in polys[p]])
                    y,x = slicing
                    xx = xx*self.dr[x] - self.r0[x]
                    yy = yy*self.dr[y] - self.r0[y]
                    polygons.append(Polygon(zip(xx,yy),
                                            facecolor=(0.0,0.8,0.2),
                                            alpha=0.4))

                self.slice_patches[idx].append(polygons)

    #-------------------------------------------------------------------------
    def rediculous_handler(self, cbox):
        mode = cbox.get_active()==0 and "enable" or "disable"
        self.connectCrosshairEvents(mode=mode)

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
        dimlengths = self.dr * N.array(self.image.shape)
        slicing = [AX, COR, SAG]
        r_center = N.array(self.zyx_coords())
        if self.zoom:
            r_neg = r_center - N.array([self.zoom/2.]*3)
            r_pos = r_center + N.array([self.zoom/2.]*3)
            self.dimlengths = N.array([self.zoom]*3)
            
        else:
            self.dimlengths = dimlengths
            r_neg = -N.take(dimlengths, slicing)/2.
            r_pos = N.take(dimlengths, slicing)/2.
            
        ## this could be made nicer?
        self.sliceplots[0].setXYlim( (r_neg[2], r_pos[2]),
                                      (r_neg[1], r_pos[1]) )
        self.sliceplots[1].setXYlim( (r_neg[2], r_pos[2]),
                                       (r_neg[0], r_pos[0]) )
        self.sliceplots[2].setXYlim( (r_neg[1], r_pos[1]),
                                       (r_neg[0], r_pos[0]) )
        self.updateSlices()
        self.setUpAxesSize()
        self.sliceplots[0].setCrosshairs(r_center[2],r_center[1])
        self.sliceplots[1].setCrosshairs(r_center[2],r_center[0])
        self.sliceplots[2].setCrosshairs(r_center[1],r_center[0])

    #-------------------------------------------------------------------------
    def _create_action_group(self):
        entries = (
            ( "FileMenu", None, "_File" ),
            ( "Quit", gtk.STOCK_QUIT,
              "_Quit", "<control>Q",
              "Quits",
              lambda action: self.destroy() ),
            ( "ToolsMenu", None, "_Tools" ),
            ( "VOI", None, "_VOI", "<control>V", "VOIgrab",
              self.VOI_handler ),
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
        xformlabel = gtk.Label("Dir Cos: \n" + \
                               str(image.orientation_xform.tomatrix()))
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
        return "%d x %d x %d"%image.shape
    #-------------------------------------------------------------------------
    def getvox(self, image):
        return "%1.3f x %1.3f x %1.3f"%(image.zsize, image.ysize, image.xsize)
    #-------------------------------------------------------------------------
    def getorigin(self, image):
        return "%d x %d x %d"%(image.z0/image.zsize,
                               image.y0/image.ysize,
                               image.x0/image.xsize)
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

    def __init__(self, vox_coords, zyx_coords):
        main_table = gtk.Table(1,2)
        main_table.attach(gtk.Label(""), 0, 1, 1, 2)

        subframe = gtk.Frame()
        subframe.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        subframe.set_size_request(300,170)

        sub_table = gtk.Table(4,2)
        sub_table.attach(gtk.Label("MEG:"), 0, 1, 0, 1)
        sub_table.attach(gtk.Label("mm:"), 0, 1, 1, 2)
        sub_table.attach(gtk.Label("vx:"), 0, 1, 2, 3)
        sub_table.attach(gtk.Label("MNI:"), 0, 1, 3, 4)
        self.meg_loc = gtk.Label("??  ?? ??")
        self.zyx_loc = gtk.Label("%2.1f %2.1f %2.1f"%zyx_coords)
        self.vx_loc = gtk.Label("%d %d %d"%vox_coords)
        self.mni_loc = gtk.Label("?? ?? ??")
        sub_table.attach(self.meg_loc, 1, 2, 0, 1)
        sub_table.attach(self.zyx_loc, 1, 2, 1, 2)
        sub_table.attach(self.vx_loc, 1, 2, 2, 3)
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

###############################################################################
################## MATPLOTLIB HACKS! ##########################################
###############################################################################
from matplotlib.patches import Rectangle
class MyLasso(Widget):
    def __init__(self, ax, xy, callback=None, useblit=True):
        self.axes = ax
        self.figure = ax.figure
        self.canvas = self.figure.canvas
        self.useblit = useblit
        if useblit:
            self.background = self.canvas.copy_from_bbox(self.axes.bbox)

        x, y = xy
        self.verts = [(x,y)]
        self.line = Line2D([x], [y], linestyle='-', color='purple', lw=2)
        self.axes.add_line(self.line)
        self.callback = callback
        self.cids = []
        self.cids.append(self.canvas.mpl_connect('button_release_event', self.onrelease))
        self.cids.append(self.canvas.mpl_connect('motion_notify_event', self.onmove))

    #-------------------------------------------------------------------------
    def onrelease(self, event):
        if self.verts is not None:
            self.verts.append((event.xdata, event.ydata))
            if len(self.verts)>2:
                self.callback(self.verts)
            self.axes.lines.remove(self.line)
        self.verts = None
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)

    #-------------------------------------------------------------------------
    def onmove(self, event):
        if self.verts is None: return 
        if event.inaxes != self.axes: return
        if event.button!=1: return 
        self.verts.append((event.xdata, event.ydata))

        self.line.set_data(zip(*self.verts))

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.axes.draw_artist(self.line)
            self.canvas.blit(self.axes.bbox)
        else:
            self.canvas.draw_idle()


class MyPolyDraw(Widget):
    def __init__(self, ax, xy, callback=None, useblit=True):
        self.axes = ax
        self.figure = ax.figure
        self.canvas = self.figure.canvas
        self.useblit = useblit
        if useblit:
            self.background = self.canvas.copy_from_bbox(self.axes.bbox)

        x, y = xy
        self.verts = [(x,y)]
        self.line = Line2D([x], [y], linestyle='-', color='purple', lw=2)
        self.axes.add_line(self.line)
        self.callback = callback
        self.cids = []
        self.cids.append(self.canvas.mpl_connect('button_release_event', lambda e: e))
        self.cids.append(self.canvas.mpl_connect('button_press_event', self.addvertex))
        self.cids.append(self.canvas.mpl_connect('motion_notify_event', self.onmove))

    #-------------------------------------------------------------------------
    def onrelease(self, event):
        if self.verts is not None:
            self.verts.append((event.xdata, event.ydata))
            if len(self.verts)>2:
                self.callback(self.verts)
            self.axes.lines.remove(self.line)
        self.verts = None
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)

    #-------------------------------------------------------------------------
    def addvertex(self, event):
        if event.inaxes != self.axes or self.verts is None:
            return
        if event.button != 1:
            self.onrelease(event)
            return
        self.verts.append((event.xdata, event.ydata))
        x,y = zip(*self.verts)
        x = (event.xdata,) + x
        y = (event.ydata,) + y
        self.line.set_data(x,y)
        if self.useblit:
            self.canvas.restore_region(self.background)
            self.axes.draw_artist(self.line)
            self.canvas.blit(self.axes.bbox)
        else:
            self.canvas.draw_idle()
        
    #-------------------------------------------------------------------------
    def onmove(self, event):
        if self.verts is None: return 
        if event.inaxes != self.axes: return
        x,y = zip(*self.verts)
        x = (event.xdata,) + x + (event.xdata,)
        y = (event.ydata,) + y + (event.ydata,)
        self.line.set_data(x, y)

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.axes.draw_artist(self.line)
            self.canvas.blit(self.axes.bbox)
        else:
            self.canvas.draw_idle()

###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":
    fname = sys.argv[1]
    ftype = len(sys.argv) > 2 and sys.argv[2] or "analyze"
    img = readImage(fname, ftype)
    spmclone(img)
    
