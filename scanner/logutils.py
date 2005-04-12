logfilename = "scannerlog"
#logfilename = "/export/home/despo/scannerlog"


##############################################################################
class ScannerLogEntry (object):

    #-------------------------------------------------------------------------
    def __init__( self, **kwargs ):
            for name,val in kwargs: self.__dict__[name] = val


    #-------------------------------------------------------------------------
    def __str__( self ):
        items = ["%s=%s" % (name,`val`) for name,val in self.__dict__.items()]
        items = ", ".join( items )
        return "%s( %s )" % (self.__class__.__name__, items)


    #-------------------------------------------------------------------------
    def __repr__( self ):
        return self.__str__()
            

##############################################################################
class FileLogger:

    #-------------------------------------------------------------------------
    def __init__( self, logfilename ):
        self._logfilename = logfilename

    #-------------------------------------------------------------------------
    def log( self, entry ):
        import time
        logfile = file( logfilename, "a+" )
        print >> logfile, "%s\t%s\n" %(time.asctime(), entry)
        logfile.close()


##############################################################################
class DBLogger:

    #-------------------------------------------------------------------------
    def log( self, entry ):
        # open db connection
        # insert log record to db
        # commit
        # close db connection
        pass


##############################################################################
class ScannerLogCollector (object):
    fields = [
        ("username","Your name:"),
        ("bicnum","Your BIC number:"),
        ("cphsnum","CPHS number for this session:"),
        ("pi","Principal Investigator for the CPHS:"),
        ("subjname","Subject name:"),
        ("exptdesc","Brief description of the experiment:"),
    ]

    #-------------------------------------------------------------------------
    def __init__( self ):
        self._logger = FileLogger( logfilename )
        self.entry = ScannerLogEntry()
        self.badfields = []

    #-------------------------------------------------------------------------
    def log( self, entry ):
            self._logger.log( entry )

    #-------------------------------------------------------------------------
    def run( self ):
        "to be overloaded by subclasses"
        pass

