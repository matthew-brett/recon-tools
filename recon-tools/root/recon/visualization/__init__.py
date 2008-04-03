# now visualization is a module ...
import numpy as N
import gtk
def iscomplex(a): return a.dtype.kind is 'c'

# Transforms for viewing different aspects of complex data
def ident_xform(data): return data
def abs_xform(data): return N.abs(data)
def phs_xform(data): return N.angle(data)
def real_xform(data): return data.real
def imag_xform(data): return data.imag

# utility functions
def ask_fname(parent, prompt, action="save", filter=None):
    mode = {
        "save": gtk.FILE_CHOOSER_ACTION_SAVE,
        "open": gtk.FILE_CHOOSER_ACTION_OPEN,
        "folder": gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER,
        }.get(action)
    dialog = gtk.FileChooserDialog(
        title=prompt,
        action=mode,
        parent=parent,
        buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,
                 gtk.STOCK_OK,gtk.RESPONSE_OK)
        )
    if filter:
        if type(filter) is type([]):
            for f in filter:
                dialog.add_filter(f)
        else:
            dialog.add_filter(filter)
    response = dialog.run()
    if response == gtk.RESPONSE_CANCEL:
        dialog.destroy()
        return
    fname = dialog.get_filename()
    dialog.destroy()
    return fname
    
