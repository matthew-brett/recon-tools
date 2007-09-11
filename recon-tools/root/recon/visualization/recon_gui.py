import gtk
import gobject
import os
#import pylab as P

from recon.operations import OperationManager, Parameter, Operation
from recon import imageio
from recon.visualization import ask_fname
from recon.visualization.spmclone import spmclone
from recon.visualization.sliceview import sliceview

ui_info = \
'''<ui>
   <menubar name='MenuBar'>
     <menu action='FileMenu'>
       <menuitem action='Load Image'/>
       <separator/>
       <menuitem action='Quit'/>
     </menu>
     <menu action='PlotterMenu'>
       <menuitem action='sliceview'/>
       <menuitem action='ortho plotter'/>
       <menuitem action='plotting off'/>
     </menu>
   </menubar>
</ui>'''
        

class recon_gui (gtk.Window):
    "A GUI driven reconstruction tool"

    def __init__(self, image=None, parent=None):
        gtk.Window.__init__(self)
        try:
            self.set_screen(parent.get_screen())
            self.plotter = parent
        except AttributeError:
            self.connect("destroy", lambda x: gtk.main_quit())
            self.plotter = None
        table = gtk.Table(2, 2)
        vbox = gtk.VBox(spacing=5)
        hbox = gtk.HBox(spacing=5)
        hbox.set_homogeneous(False)
        self.pnames_vbox = gtk.VBox()
        self.pvals_vbox = gtk.VBox()
        self.op_man = OperationManager()
        self.op_names = self.op_man.getOperationNames()
        self.cookOps()
        self.image = image
        # need to copy the image somehow?
        self.first_image = cheap_copy(self.image)
        self.last_image = None


        self.op_select = gtk.combo_box_new_text()
        for name in self.op_names: self.op_select.append_text(name)
        self.op_select.set_active(1)
        self.op_select.connect("changed", self.op_changed)

        self.run_op = gtk.Button(label="Run")
        #self.run_op.connect("clicked", self.hide_buttons)
        self.run_op.connect("clicked", self.op_runner)
        self.revert_op = gtk.Button(label="Revert Op")
        self.revert_op.connect("clicked", self.revert_image)
        self.original_img = gtk.Button(label="Revert All")
        self.original_img.connect("clicked", self.restore_original)
        self.buttons = [self.run_op, self.revert_op, self.original_img]

        vbox.pack_start(self.op_select)
        vbox.pack_start(self.run_op)
        vbox.pack_start(self.revert_op)
        vbox.pack_start(self.original_img)
        self.update_params(self.op_names[self.op_select.get_active()])
        hbox.pack_start(self.pnames_vbox, expand=True, fill=True)
        hbox.pack_start(self.pvals_vbox)
        vbox.set_size_request(200,150)
        hbox.set_size_request(200,150)

        merge = gtk.UIManager()
        merge.insert_action_group(self._create_action_group(), 0)
        mergeid = merge.add_ui_from_string(ui_info)
        self.menubar = merge.get_widget("/MenuBar")
        self.menubar.set_size_request(400,30)
        
        table.attach(vbox, 0, 1, 1, 2)
        table.attach(hbox, 1, 2, 1, 2)
        table.attach(self.menubar, 0, 2, 0, 1)
        table.set_col_spacing(0,5)
        self.set_default_size(400,180)
        self.set_border_width(3)
        self.add(table)
        self.set_title("Graphical Recon Tools")
        self.show_all()
        if not parent:
            gtk.main()

    def cookOps(self):
        ### remove some troublesome Ops ####
        self.op_names.remove("ReadImage")
        self.op_names.remove("RotPlane")
        self.op_names.remove("FlipSlices")
        ### insert some convenience Ops ####
        ksp_ops = ["GeometricUndistortionK", "BalPhaseCorrection",
                   "UnbalPhaseCorrection", "FillHalfSpace"]
        for op in ksp_ops:
            idx = self.op_names.index(op)
            self.op_names.insert(idx, op+"**")
        

    def update_params(self, op_name):
        op = self.op_man.getOperation(op_name)
        namebox, valbox = self.pnames_vbox, self.pvals_vbox
        for child1,child2 in zip(namebox.get_children(), valbox.get_children()):
            namebox.remove(child1)
            valbox.remove(child2)

        for n, param in enumerate(op.params):
            namebox.pack_start(gtk.Label(param.name))
            entry = gtk.Entry()
            entry.set_text(str(param.default))
            entry.set_width_chars(12)
            valbox.pack_start(entry)

    def get_params(self):
        namebox, valbox = self.pnames_vbox, self.pvals_vbox
        params = {}
        for name,val in zip(namebox.get_children(), valbox.get_children()):
            params[name.get_text()] = val.get_text()
        return params
        
    def op_changed(self, cbox):
        self.update_params(self.op_names[cbox.get_active()].strip("**"))
        self.show_all()

    def op_runner(self, button):
        self.last_image = cheap_copy(self.image)
        putative_ops = self.op_names[self.op_select.get_active()].split("**")
        op_name = putative_ops.pop(0)
        ops = [self.op_man.getOperation(op_name)(**self.get_params())]
        if putative_ops:
            # there are no params for the FTs
            ft = self.op_man.getOperation("ForwardFFT")()
            ift = self.op_man.getOperation("InverseFFT")()
            ops = [ft] + ops + [ift]
        for op in ops:
            op.log("running")
            success = op.run(self.image) or 1
        if success is not -1:
            self.update_plotter()
        else:
            print "Warning: Operation failed"
##         for button in self.buttons:
##             button.set_sensitive(True)

    def hide_buttons(self, button):
        for b in self.buttons: b.set_sensitive(False)
        print "buttons hidden"
        self.queue_draw()

    def revert_image(self, button):
        if self.last_image:
            print "reverting"            
            self.image = cheap_copy(self.last_image)
            self.last_image = None
            self.update_plotter()

    def restore_original(self, button):
        self.image = cheap_copy(self.first_image)
        self.update_plotter()

    def update_plotter(self):
        if self.plotter:
            self.plotter.externalUpdate(self.image)

    def load_new_image(self, action):
        image_filter = gtk.FileFilter()
        image_filter.add_pattern("*.fid")
        image_filter.set_name("FID Images")
        fname = ask_fname(self, "Choose file to load...", action="folder",
                          filter=image_filter)
        if not fname:
            return
        self.image = imageio.readImage(fname, "fid", vrange=(0,1))
        self.first_image = cheap_copy(self.image)
        self.last_image = None
        basedir = fname.strip(fname.split('/')[-1])
        os.chdir(basedir)
        self.update_plotter()

    def plot_handler(self, action, current):
        new_plotter = {1: sliceview,
                       2: spmclone}.get(current.get_current_value())
        if self.plotter:
            self.plotter.destroy()
        if new_plotter:
            self.plotter = new_plotter(self.image, parent=self)

    def _create_action_group(self):
        entries = (
            ( "FileMenu", None, "_File" ),
            ( "Load Image", gtk.STOCK_OPEN, "_Load Image",
              "<control>l", "Loads a new FID image to process",
              self.load_new_image ),
            ( "Quit", gtk.STOCK_QUIT, "_Quit", "<control>q",
              "Quits", lambda action: self.destroy() ),
            ( "PlotterMenu", None, "_Plotter Menu" ),
        )

        if not self.plotter:
            active_plot = 0
        elif isinstance(self.plotter, sliceview):
            active_plot = 1
        elif isinstance(self.plotter, spmclone):
            active_plot = 2
        
        plotting_toggles = (
            ( "sliceview", None, "_sliceview", None, "", 1),
            ( "ortho plotter", None, "_ortho plotter", None, "", 2),
            ( "plotting off", None, "_plotting off", None, "", 0),
        )
            
        action_group = gtk.ActionGroup("WindowActions")
        action_group.add_actions(entries)
        action_group.add_radio_actions(plotting_toggles, active_plot,
                                       self.plot_handler)

        # if launched from a plotter, don't give the user the chance
        # to destroy that plotter from here; so disable these toggles
        if active_plot:
            action_group.get_action("Load Image").set_sensitive(False)
            for plot_entry in plotting_toggles:
                action_group.get_action(plot_entry[0]).set_sensitive(False)
            
        
        return action_group
        


def cheap_copy(src, copy_array=True):
    if not src:
        return None
    dest = imageio.ReconImage(src[:], src.isize, src.jsize,
                              src.ksize, src.tsize)
    for item in src.__dict__.items():
        dest.__dict__[item[0]] = item[1]
    dest.__class__ = src.__class__
    if copy_array:
        dest.data = src.data.copy()
    return dest
