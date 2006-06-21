
import os
from StringIO import StringIO
from optparse import OptionParser
from imaging.tools.Recon import Recon
from imaging.tools import OrderedConfigParser
from imaging.operations import OperationManager, RunLogger

class Rerun (Recon):

    _opmanager = OperationManager()
    #-------------------------------------------------------------------------
    def __init__(self, logFile):
        self.logFile = os.path.abspath(logFile)
        OptionParser.__init__(self)  # don't expect any args
        self.add_options(self.options)
        
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
    def configureOperations(self, opfileptr):
        """
        Overloads Recon's method by using the readfp() method, otherwise
        performs identically.
        """
        config = OrderedConfigParser()
        config.readfp(opfileptr)
        opname = lambda k: k.rsplit(".",1)[0]
        return [
          (self._opmanager.getOperation(opname(opkey)), dict(config.items(opkey)))
          for opkey in config.sections()]
    
    #-------------------------------------------------------------------------
    def getOptions(self):
        options, args = self.parse_args()
        configStr = self._cleanLog()
        configFP = StringIO(configStr)
        options.operations = self.configureOperations(configFP)
        options.log_file = "/dev/null"  # no need to log the re-run to file
        self.confirmOps(options.operations)
        return options
        
        

    
        
    
