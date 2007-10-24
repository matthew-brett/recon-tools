
import os
from StringIO import StringIO
from optparse import OptionParser
from recon.tools.Recon import Recon
from recon.tools import OrderedConfigParser
from recon.operations import OperationManager, RunLogger

class Rerun (Recon):

    _opmanager = OperationManager()
    #-------------------------------------------------------------------------
    def __init__(self, logFile):
        self.logFile = os.path.abspath(logFile)
        Recon.__init__(self)        
    #-------------------------------------------------------------------------
    def _cleanLog(self):
        startkey = RunLogger(file('/dev/null', 'w'))._start_string
        log = file(self.logFile)
        ops_str = ""
        read = True
        while log.readline() != startkey:
            pass
        while read==True:
            try:
                # remove the leading hash char (#)
                ops_str += log.next()[1:]
            except StopIteration:
                read = False
        return ops_str
    #-------------------------------------------------------------------------
    def getOptions(self):
        options, args = self.parse_args()
        configStr = self._cleanLog()
        configFP = StringIO(configStr)
        options.operations = self.configureOperations(configFP)
        options.log_file = "/dev/null"  # no need to log the re-run to file
        self.confirmOps(options.operations)
        return options
        
        

    
        
    
