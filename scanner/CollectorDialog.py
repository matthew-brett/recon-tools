from wxPython.wx import *
from logutils import *


#############################################################################
class CollectorDialog (wxDialog):

    #-------------------------------------------------------------------------
    def initialize( self, entry, badfields=None ):
        self.entry = entry
        self.badfields = badfields or []
        self.ctrls = {}

        # Now continue with the normal construction of the dialog
        # contents
        sizer = wxBoxSizer( wxVERTICAL )
        self.sizer = sizer

        # if some fields were invalid, show a message
        if self.badfields:
            box = wxBoxSizer( wxHORIZONTAL )
            label = wxStaticText( self, -1, r'"Please fill in the fields marked with *.' )
            label.SetForegroundColour( wxRED )
            box.Add( label, 0, wxALIGN_CENTRE|wxALL, 5 )
            self.sizer.AddSizer( box, 0, wxGROW|wxALIGN_CENTER_VERTICAL|wxALL, 5 )

        # add a text box for each field
        for fieldname,prompt in ScannerLogCollector.fields:
            self.addField( fieldname, prompt )

        line = wxStaticLine( self, -1, size=(20,-1), style=wxLI_HORIZONTAL )
        sizer.Add( line, 0, wxGROW|wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP, 5 )

        box = wxBoxSizer( wxHORIZONTAL )
        btn = wxButton( self, wxID_OK, " OK " )
        btn.SetDefault()
        box.Add( btn, 0, wxALIGN_CENTRE|wxALL, 5 )
        sizer.AddSizer( box, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 )

        self.SetSizer( sizer )
        self.SetAutoLayout( True )
        sizer.Fit( self )


    #-------------------------------------------------------------------------
    def addField( self, name, label ):
        box = wxBoxSizer(wxHORIZONTAL)

        # mark fields which need to be changed
        label = wxStaticText(self, -1, label)
        box.Add(label, 0, wxALIGN_CENTRE|wxALL, 5)

        if name in self.badfields:
            label = wxStaticText(self, -1, "*")
            label.SetForegroundColour( wxRED )
            box.Add(label, 0, wxALIGN_CENTRE|wxALL, 5)
        
        value = str( getattr( self.entry, name, "" ) )
        textctrl = wxTextCtrl( self, -1, value, size=(80,-1) ) 
        box.Add(textctrl, 1, wxALIGN_CENTRE|wxALL, 5)

        self.sizer.AddSizer(box, 0, wxGROW|wxALIGN_CENTER_VERTICAL|wxALL, 5)
        self.ctrls[name] = textctrl


    #-------------------------------------------------------------------------
    def getValues( self ):
        self.entry.__dict__.update( dict( [(n, t.GetValue()) for n, t in self.ctrls.items()] ) )


##############################################################################
if __name__ == "__main__":
    app = wxPySimpleApp( 0 )
    collect_dialog = CollectorDialog( None, -1, "Scanner Usage Log",
        size=wxSize(350, 200), style=wxDEFAULT_DIALOG_STYLE )
    collect_dialog.initialize( ScannerLogEntry() )
    collect_dialog.CenterOnScreen()
    val = collect_dialog.ShowModal()
    if val == wxID_OK:
        collect_dialog.getValues()
    print collect_dialog.entry


