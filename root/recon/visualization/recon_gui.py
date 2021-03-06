import gtk
import gobject
import os

from recon.operations import OperationManager, Parameter, Operation
from recon.imageio import readImage
from recon.util import TempMemmapArray
from recon.visualization import ask_fname
from recon.visualization.spmclone import spmclone
from recon.visualization.sliceview import sliceview

# TODO
# * implement logging??

def get_toplevel_selected(tree):
    """
    Given a tree, returns the top level name of the selected row
    and the list model in which it is embedded.
    """
    model, row = tree.get_selection().get_selected()
    if row is not None:
        while model.iter_parent(row) is not None:
            row = model.iter_parent(row)
    return (model, row)

ui_info = \
'''<ui>
   <menubar name='MenuBar'>
     <menu action='FileMenu'>
       <menuitem action='Load Image'/>
       <separator/>
       <menuitem action='Quit'/>
     </menu>
     <menu action='Tools'>
       <menuitem action='Default oplist'/>
     </menu>
     <menu action='PlotterMenu'>
       <menuitem action='sliceview'/>
       <menuitem action='ortho plotter'/>
       <menuitem action='plotting off'/>
     </menu>
   </menubar>
</ui>'''

class recon_gui (gtk.Window):
    "Provides a graphical environment to construct and test oplists."

    def __init__(self, image=None, logfile=None, parent=None):
        gtk.Window.__init__(self)
        try:
            self.set_screen(parent.get_screen())
            self.plotter = parent
        except AttributeError:
            self.connect('destroy', lambda x: gtk.main_quit())
            self.plotter = None

        self.image = image
        self._initialize_undo_buffer()
        self.pnames_vbox = gtk.VBox()
        self.pvals_vbox = gtk.VBox()
        
        win_table = gtk.Table(rows=2, columns=3)

        # Make three widget panels
        left_panel = gtk.Table(rows=7, columns=3)
        mid_panel = gtk.VBox()
        right_panel = gtk.HBox(homogeneous=False)


        ###### set up left-hand panel (lots of objects!!)
        left_panel.set_size_request(200,150)


        ###### open Button ######
        openbutton = gtk.Button(stock=gtk.STOCK_OPEN)
        openbutton.connect('clicked', self.load_any_image)
        left_panel.attach(openbutton, 0, 1, 0, 1)

        ###### path Entry ######
        self.pathentry = gtk.Entry()
        self.pathentry.set_width_chars(20)
        left_panel.attach(self.pathentry, 1, 2, 0, 1)
        if image and hasattr(image, 'path'):
            image_has_path = True
            fname = image.path
            self.pathentry.set_text(fname.split('/')[-1])
        else:
            image_has_path = False

        ###### reload Button ######
        reloadbutton = gtk.Button(stock=gtk.STOCK_REFRESH)
        reloadbutton.connect('clicked', self.reload_image)
        left_panel.attach(reloadbutton, 2, 3, 0, 1)
        if not image_has_path:
            reloadbutton.set_sensitive(False)

        ###### row Separator ######
        left_panel.attach(gtk.SeparatorToolItem(), 0, 3, 1, 2)

        ###### move-op-up Button ######
        upbutton = gtk.Button(stock=gtk.STOCK_GO_UP)
        upbutton.connect('clicked', self.move_op_up)
        left_panel.attach(upbutton, 0, 1, 2, 3)

        ###### move-op-down Button ######
        dnbutton = gtk.Button(stock=gtk.STOCK_GO_DOWN)
        dnbutton.connect('clicked', self.move_op_dn)
        left_panel.attach(dnbutton, 0, 1, 3, 4)

        ###### oplist TreeView and ScrolledWindow ######
        sw_oplist = gtk.ScrolledWindow()
        sw_oplist.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        # the TreeStore is a list widget that holds the op entries
        self.ops_store = gtk.TreeStore(gobject.TYPE_STRING,
                                       gobject.TYPE_BOOLEAN)
        # the TreeView widget is the top-level display object
        self.oplist_tree = gtk.TreeView(self.ops_store)
        # This handles the text rendering, and reports if the text changes..
        # since we're allowing parameters to be edited, connect a callback
        cell = gtk.CellRendererText()
        cell.connect("edited", self.param_edited, self.ops_store)
        col = gtk.TreeViewColumn('Operations', cell, text=0, editable=True)
        self.oplist_tree.append_column(col)
        sw_oplist.add(self.oplist_tree)
        left_panel.attach(sw_oplist, 1, 2, 2, 7)
        
        ###### add-op Button ######
        addbutton = gtk.Button(stock=gtk.STOCK_ADD)
        addbutton.connect('clicked', self.append_selected_op)
        left_panel.attach(addbutton, 2, 3, 2, 3)

        ###### remove-op Button ######
        rembutton = gtk.Button(stock=gtk.STOCK_REMOVE)
        rembutton.connect('clicked', self.remove_op_from_list)
        left_panel.attach(rembutton, 2, 3, 3, 4)

        ###### row Separators ######
        left_panel.attach(gtk.SeparatorToolItem(), 0, 1, 4, 5)
        left_panel.attach(gtk.SeparatorToolItem(), 2, 3, 4, 5)
        
        ###### run-oplist Button ######
        runbutton = gtk.Button(stock=gtk.STOCK_MEDIA_PLAY, label='Run oplist')
        runbutton.connect('clicked', self.run_oplist)
        left_panel.attach(runbutton, 0, 1, 5, 6)

        ###### step-back Button ######
        undobutton = gtk.Button(stock=gtk.STOCK_MEDIA_REWIND, label='Step back')
        undobutton.connect('clicked', self.revert_data)
        left_panel.attach(undobutton, 2, 3, 5, 6)

        ###### save-img Button ######
        saveimg = gtk.Button(label='Save image')
        saveimg.connect('clicked', self.save_image)
        #saveimg.set_label('Save image')
        left_panel.attach(saveimg, 0, 1, 6, 7)

        ###### save-oplist Button ######
        saveops = gtk.Button(label='Save oplist')
        saveops.connect('clicked', self.save_oplist)
        left_panel.attach(saveops, 2, 3, 6, 7)
        
        left_panel.set_size_request(400,200)

        ###### set up middle panel
        sw_ops = gtk.ScrolledWindow()
        sw_ops.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        sw_ops.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        mid_panel.pack_start(sw_ops)
        self.op_man = OperationManager()
        self.op_names = self.op_man.getOperationNames()
        self.op_tree = gtk.TreeView(self._fill_ops_model())
        self.op_tree.get_selection().set_mode(gtk.SELECTION_SINGLE)
        self.op_tree.get_selection().connect('changed', self.op_changed)        
        self._add_ops_column(self.op_tree)
        sw_ops.add(self.op_tree)
        mid_panel.set_size_request(250,200)

        ###### set up the right-hand panel
        right_panel.pack_start(self.pnames_vbox, expand=True, fill=True)
        right_panel.pack_start(self.pvals_vbox)
        right_panel.set_size_request(250,200)

        # do menubar
        merge = gtk.UIManager()
        self.action_group = self._create_action_group()
        merge.insert_action_group(self.action_group, 0)
        mergeid = merge.add_ui_from_string(ui_info)
        menubar = merge.get_widget('/MenuBar')
        menubar.set_size_request(900,30)

        win_table.attach(menubar, 0, 3, 0, 1)
        win_table.attach(left_panel, 0, 1, 1, 2)
        win_table.attach(mid_panel, 1, 2, 1, 2)
        win_table.attach(right_panel, 2, 3, 1, 2)
        win_table.set_col_spacing(0,5)
        self.set_default_size(900,230)
        self.set_border_width(3)
        self.add(win_table)
        self.set_title('Graphical Recon Tools')
        self.show_all()
        if not parent:
            gtk.main()

    def _fill_ops_model(self):
        # first kill some operations from the list ...
        
        # these are superfluous
        self.op_names.remove('ReadImage')
        self.op_names.remove('ViewOrtho')
        self.op_names.remove('ViewImage')
        self.op_names.remove('Template')
        # any op that messes with the image dimension sizes should be avoided
        self.op_names.remove('RotPlane')
        self.op_names.remove('FlipSlices')
        # this fails due to MPL holding onto the array in a weird way
        self.op_names.remove('ZeroPad')
        

        model = gtk.ListStore(gobject.TYPE_STRING, gobject.TYPE_BOOLEAN)
        for op in self.op_names:
            iter = model.append()
            model.set(iter, 0, op, 1, False)
        return model

    def _add_ops_column(self, treeview):
        model = treeview.get_model()
        renderer = gtk.CellRendererText()
        # need to connect a 'on selected' handler to pop up param panels
        renderer.set_data('column', 0)
        column = gtk.TreeViewColumn('Operations', renderer, text=0, editable=1)
        treeview.append_column(column)

    def _create_action_group(self):
        entries = (
            ( 'FileMenu', None, '_File' ),
            ( 'Load Image', gtk.STOCK_OPEN, '_Load Image',
              '<control>l', 'Loads a new FID image to process',
              self.load_any_image ),
            ( 'Quit', gtk.STOCK_QUIT, '_Quit', '<control>q',
              'Quits', lambda action: self.destroy() ),
            ( 'Tools', None, '_Tools' ),
            ( 'Default oplist', None, 'Default oplist',
              None, 'Populates oplist with a suggested default',
              self.use_default_ops ),
            ( 'PlotterMenu', None, '_Plotter Menu' ),
        )

        if not self.plotter:
            active_plot = 0
        elif isinstance(self.plotter, sliceview):
            active_plot = 1
        elif isinstance(self.plotter, spmclone):
            active_plot = 2
        
        plotting_toggles = (
            ( 'sliceview', None, '_sliceview', None, '', 1),
            ( 'ortho plotter', None, '_ortho plotter', None, '', 2),
            ( 'plotting off', None, '_plotting off', None, '', 0),
        )
            
        action_group = gtk.ActionGroup('WindowActions')
        action_group.add_actions(entries)
        action_group.add_radio_actions(plotting_toggles, active_plot,
                                       self.plot_handler)

        # if launched from a plotter, don't give the user the chance
        # to destroy that plotter from here; so disable these toggles
        if active_plot:
            action_group.get_action('Load Image').set_sensitive(False)
            for plot_entry in plotting_toggles:
                action_group.get_action(plot_entry[0]).set_sensitive(False)
        
        
        return action_group

    def _initialize_undo_buffer(self):
        if not self.image:
            self.prev_img_data = None
        else:
            if hasattr(self.image, 'cdata'):
                self.prev_img_data = TempMemmapArray(self.image.cdata.shape,
                                                     self.image.cdata.dtype)
                self.prev_img_data[:] = self.image.cdata.copy()
            else:
                self.prev_img_data = TempMemmapArray(self.image.data.shape,
                                                     self.image.data.dtype)
                self.prev_img_data[:] = self.image.data.copy()

    def _update_undo_buffer(self):
        if not self.image:
            return
        if hasattr(self.image, 'cdata'):
            self.prev_img_data[:] = self.image.cdata.copy()
        else:
            self.prev_img_data[:] = self.image.data.copy()

    def _restore_from_undo_buffer(self):
        if hasattr(self.image, 'cdata'):
            self.image.cdata[:] = self.prev_img_data[:]
            self.image.use_membuffer(0)
        else:
            self.image.data[:] = self.prev_img_data[:]

    def construct_opchain(self):
        """
        Generates a Python list of operations based on the entries in the
        oplist tree.
        """
        oplist = self.construct_oplist()
        opchain = []
        for opname, pdict in oplist:
            opchain.append(self.op_man.getOperation(opname)(**pdict))
        return opchain

    def construct_oplist(self):
        "Generates a Python list of (opname, {params-dict}) pairs"
        oplist = []
        model = self.ops_store
        oprow = model.get_iter_first()
        while oprow is not None:
            opname = model.get_value(oprow, 0)
            pdict = {}
            if model.iter_has_child(oprow):
                for n in xrange(model.iter_n_children(oprow)):
                    paramrow = model.iter_nth_child(oprow, n)
                    pstr = model.get_value(paramrow, 0)
                    pname, pval = pstr.split('=')
                    pdict[pname.strip()] = pval.strip()
            oplist.append((opname, pdict))
            oprow = model.iter_next(oprow)
        return oplist

    def get_selected_opname(self):
        "Returns the selected operation in the list of all ops"
        model, row = get_toplevel_selected(self.op_tree)
        return row is not None and model.get_value(row, 0) or None

    def get_params(self):
        """
        Returns a dictionary of parameter names and values from current
        parameter panel
        """
        namebox, valbox = self.pnames_vbox, self.pvals_vbox
        params = {}
        for name,val in zip(namebox.get_children(), valbox.get_children()):
            params[name.get_text()] = val.get_text()
        return params

    def update_params(self, op_name):
        "Changes the parameter panel to reflect given operation"
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

    def move_op(self, direction):
        "Tries to swap a selected row with an entry n=direction rows away"
        model, row = get_toplevel_selected(self.oplist_tree)
        if row is None:
            return
        path = list(model.get_path(row))
        path[0] = path[0] + direction
        try:
            swapper = model.get_iter(tuple(path))
            self.ops_store.swap(row, swapper)
        except:
            return

    def update_plotter(self):
        if self.plotter:
            self.plotter.externalUpdate(self.image)

    def append_op_to_list(self, opname, params=None, after_row=None):
        if after_row:
            row = self.ops_store.insert_after(None, after_row)
        else:
            row = self.ops_store.append(None)
        self.ops_store.set(row, 0, opname, 1, False)
        if params:
            for item, val in params.items():
                subrow = self.ops_store.append(row)
                txt = '%s = %s'%(item, val)
                self.ops_store.set(subrow, 0, txt, 1, True)
        

    ####### CALLBACK METHODS #######

    def op_changed(self, treeselection, *args):
        "called when the op selection changes in the list of available ops"
        opname = self.get_selected_opname()
        if opname is None:
            return
        self.update_params(opname)
        self.show_all()
        
    def append_selected_op(self, button):
        "called when add button is hit"
        op = self.get_selected_opname()
        if op is None:
            return
        params = self.get_params()
        # put this in to activate "insert-after-selected" mode
        #model, selected_row = get_toplevel_selected(self.oplist_tree)
        selected_row = None
        self.append_op_to_list(op, params=params, after_row=selected_row)
##         row = self.ops_store.insert_after(None, selected_row) \
##               if selected_row else self.ops_store.append(None)
##         self.ops_store.set(row, 0, op, 1, False)
##         if params:
##             for item, val in params.items():
##                 subrow = self.ops_store.append(row)
##                 txt = '%s = %s'%(item, val)
##                 self.ops_store.set(subrow, 0, txt, 1, True)

    def remove_op_from_list(self, button):
        "called when remove button is hit"
        model, row = get_toplevel_selected(self.oplist_tree)
        if row is None:
            return
        path = list(model.get_path(row))[0]
        model.remove(row)
        # if there's a row at path, select it
        # else if there's a row at path-1, select it
        # else do nothing (no ops left)
        while path >= 0:
            try:
                model.get_iter((path,))
                self.oplist_tree.set_cursor((path,))
                break
            except:
                path -= 1
        
    def move_op_up(self, button):
        "called when up button is hit"
        self.move_op(-1)

    def move_op_dn(self, button):
        "called when down button is hit"
        self.move_op(+1)

    def param_edited(self, cell, path_str, new_text, model):
        "called when parameter's value is edited in current oplist"
        row = model.get_iter_from_string(path_str)
        model.set(row, 0, new_text, 1, True)

    def save_oplist(self, button):
        "called when save oplist button is hit"
        fname = ask_fname(self, 'Write oplist to this file...', action='save')
        if not fname:
            return
        oplist = self.construct_oplist()
        f = open(fname, 'w')
        for opdef in oplist:
            f.write('['+ opdef[0] + ']\n')
            for key,val in opdef[1].items():
                f.write('%s = %s\n'%(key, val))
            f.write('\n')
        f.close()

    def save_image(self, button):
        "called when save image button is hit"
        if not self.image:
            return
        image_filter = gtk.FileFilter()
        image_filter.add_pattern('*.hdr')
        image_filter.set_name('ANALYZE files')
        fname = ask_fname(self, 'Write image to this file...', action='save',
                          filter=image_filter)
        if not fname:
            return
        if not self.image.combined:
            self.image.combine_channels()
        self.image.writeImage(fname, targetdim=self.image.ndim)


    def run_oplist(self, button):
        "called when run oplist (play) button is hit"
        if not self.image:
            return
        self._update_undo_buffer()
        opchain = self.construct_opchain()
        self.image.runOperations(opchain)
        self.update_plotter()

    def revert_data(self, button):
        "called when go-back (rewind) button is hit"
        if not self.image:
            return
        self._restore_from_undo_buffer()
        self.update_plotter()

    def use_default_ops(self, action):
        print "this actually may not be a good idea"

    def load_any_image(self, action):
        "called when open button/menu item is hit"
        fid_filter = gtk.FileFilter()
        fid_filter.add_pattern('fid')
        fid_filter.set_name('FID Images')

        dat_filter = gtk.FileFilter()
        dat_filter.add_pattern('*.dat')
        dat_filter.set_name('DAT Images')

        ana_nii_filter = gtk.FileFilter()
        ana_nii_filter.add_pattern('*.hdr')
        ana_nii_filter.add_pattern('*.nii')
        ana_nii_filter.set_name('ANALYZE/NIFTI Images')

        fname = ask_fname(self, 'Choose file to load...', action='open',
                          filter=[fid_filter, dat_filter, ana_nii_filter])

        if not fname:
            return
        # cook any fid filename found
        if fname[-8:] == '.fid/fid':
            fname = os.path.split(fname)[0]
        self.image = readImage(fname, vrange=(0,1))
        self._initialize_undo_buffer()
        self.pathentry.set_text(fname.split('/')[-1])
        self.update_plotter()

    def reload_image(self, button):
        "called when reload (refresh) button is hit"
        if not self.image:
            return
        path = self.image.path
        self.image = readImage(path, vrange=(0,1))
        self._initialize_undo_buffer()
        self.update_plotter()

    def plot_handler(self, action, current):
        "switches plotting modes"
        new_plotter = {1: sliceview,
                       2: spmclone}.get(current.get_current_value())

        if self.plotter:
            self.plotter.handler_block(self.plotter.destroy_handle)
            self.plotter.destroy()
            self.plotter = None

        if self.image and new_plotter:
            self.plotter = new_plotter(self.image, parent=self)
            
    def _plotter_died(self):
        "called when the external plotting window is closed"
        self.action_group.get_action('plotting off').activate()
