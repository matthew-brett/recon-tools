from logutils import ScannerLogCollector


############################################################################
class WXScannerLogCollector (ScannerLogCollector):

    #------------------------------------------------------------------------
    def run( self ):
        from wxPython.wx import wxPySimpleApp
        app = wxPySimpleApp( 0 )

        # collect until correct
        self.collectInfo()
        while not self.validate():
            self.collectInfo()

        # log it
        self.log( self.entry )


    #------------------------------------------------------------------------
    def collectInfo( self ):
        from wxPython.wx import wxDEFAULT_DIALOG_STYLE, wxSize, wxID_OK
        from CollectorDialog import CollectorDialog

        collect_dialog = CollectorDialog( None, -1, "Scanner Usage Log",
            size=wxSize(350, 200), style=wxDEFAULT_DIALOG_STYLE )
        collect_dialog.initialize( self.entry, self.badfields )
        collect_dialog.CenterOnScreen()
        collect_dialog.ShowModal()
        collect_dialog.getValues()


    #------------------------------------------------------------------------
    def validate( self ):
        self.badfields = []
        for fieldname, prompt in self.fields:
            if fieldname is "exptdesc": continue
            value = getattr( self.entry, fieldname, None )
            if not value:
                self.badfields.append( fieldname )
                
        return not self.badfields



