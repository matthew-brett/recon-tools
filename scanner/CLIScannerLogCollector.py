#############################################################################
class CLIScannerLogCollector (object):

    #------------------------------------------------------------------------
    def collectInfo( self ):
        for fieldname,prompt in self.fields:
            setattr( self.entry, fieldname, raw_input( prompt ) )


    #------------------------------------------------------------------------
    def validate( self ):
        # check user name in password file
        # check BIC num in list of valid BIC nums
        # check CPHS num in list of valid CPHS nums
        # check Principal Investigator name
        pass


    #------------------------------------------------------------------------
    def verify( self ):
        import sys
        print "You entered:"
        for fieldname,prompt in self.fields:
            print "\t%s" % prompt, getattr( self.entry, fieldname )

        sys.stdout.write( "\nIs this correct?  (y)es (n)o  " )
        char = sys.stdin.read(1)
        while char not in ['y', 'Y', 'n', 'N']:
            char = sys.stdin.read(1)
        sys.stdin.readline()

        if char in ['y', 'Y']: return "YES"
        if char in ['n', 'N']: return "NO"


    #------------------------------------------------------------------------
    def run( self ):
        
        # collect until correct
        self.collectInfo()
        while self.verify() == "NO":
            self.collectInfo()

        # log it
        self.log( self.entry )
        
        print "Thank you!"
        print "Launching VNMR..."

   


