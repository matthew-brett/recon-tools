"Defines a command-line interface to the recon tool."
from optparse import OptionParser, Option
import os, sys
import recon.conf
import recon
from recon.imageio import available_writers as output_format_choices, \
     output_datatypes as output_datatype_choices
from recon.tools import OrderedConfigParser, ConsoleTool, default_logfile, \
     parseVolRangeOption
from recon.operations import OperationManager, RunLogger
from recon.scanners.varian import getPulseSeq

# here is a global flag to control fast/slow array logic
_FAST_ARRAY = False

class NoArgsException (Exception):
    pass

##############################################################################
class Recon (ConsoleTool):
    """
    Handle command-line aspects of the recon tool.
    @cvar options: tuple of Option objs, filled in by OptionParser
    """

    _opmanager = OperationManager()
    default_logfile = "recon.log"

    options = (
        # reserving obvious -o for --outfile or something
      Option("-p", "--oplist", dest="oplist", type="string",
        default=None, action="store",
        help="Name of the oplist file describing operations and parameters."),

      Option("-r", "--vol-range", dest="vol_range", type="string",
        default=":", action="store",
        help="Which image volumes to reconstruct. Format is start:end, "\
        "where either start or end may be omitted, indicating to start "\
        "with the first or end with the last respectively. The index of "\
        "the first volume is 0. The default value is a single colon "\
        "with no start or end specified, meaning process all image volumes."),

      Option("-f", "--file-format", dest="file_format", action="store",
        type="choice", default=output_format_choices[0],
        choices=output_format_choices,
        help="{%s} "\
        "analyze: Save individual image for each frame in analyze format. "\
        "nifti-dual: save nifti file in (hdr, img) pair. "\
        "nifti-single: save nifti file in single-file format."%\
             (" | ".join(output_format_choices))),

      Option("-y", "--output-data-type", dest="output_datatype",
        type="choice", default=output_datatype_choices[0], action="store",
        choices=output_datatype_choices,
        help="""{%s}
        Specifies whether output images should contain only magnitude or
        both the real and imaginary components."""%\
             (" | ".join(output_datatype_choices))),

      Option("-l", "--log-file", default=default_logfile,
        help="where to record reconstruction details ('%s' by default)"\
             %default_logfile),

      Option("-x", action="store_true", dest="fastArray", default=False,
             help="this may shave off a few seconds from the reconstruction, "\
             "but expect memory usage to go up by a factor of 4."),

      Option("-s", "--suffix", action="store", default=None,
             help="Overrides the default naming of output files."),

      Option("-n", "--filedim", action="store", default=3,
             help="Sets the number of dimensions per output file "\
             "(defaults to 3)"),

      Option("-u", "--opusage", dest="ophelp", action="store", default=None,
             help="gives info on an operation and its parameters"),

      Option("-e", "--opsexample", dest="opsexample", default=False,
             action="store_true",
             help="For a given fid, print an example oplist to the screen. "\
             "Usage: recon -e[--opsexample] somedata.fid"))

      

    #-------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        v = "ReconTools " + recon.__version__
        OptionParser.__init__(self, version=v, *args, **kwargs)
        self.set_usage("usage: %prog [options] datadir outfile")
        self.add_options(self.options)

    #-------------------------------------------------------------------------
    def _running(self, opname, opseq):
        # see if current Recon is running said operation
        ops = [opclass for (opclass, _) in opseq]
        return self._opmanager.getOperation(opname) in ops
    
    #-------------------------------------------------------------------------
    def _argsFromPWD(self):
        # try to get correct files from pwd or throttle
        if os.path.isfile("fid") and os.path.isfile("procpar"):
            pwd = os.path.abspath(".")
            name = os.path.split(pwd[:pwd.rfind(".fid")])[-1]
            return (pwd, name+".recon")
        else:
            raise NoArgsException

    #-------------------------------------------------------------------------
    def _printOpHelp(self, opname):
        if opname not in self._opmanager.getOperationNames():
            print "There is no operation named %s"%opname
        else:
            print "help for %s"%opname            
            self._opmanager.getOperation(opname)().opHelp()

    #-------------------------------------------------------------------------
    def _printOplistEx(self, oplist):
        ops = open(oplist).read()
        print "\nBasic reconstruction sequence (copy & paste into oplist):\n"
        print ops
        print "For instructions for a given op, run recon -u(--opusage) opname"
        
    #-------------------------------------------------------------------------
    def _findOplist(self, datadir):
        "Determine which stock oplist to use based on pulse sequence."
        
        oplistBySeq = {
          "epidw": "epi.ops",
          "epi":   "epi.ops",
          "gems":  "gems.ops",
          "mp_flash3d0": "mpflash3D.ops",
          "mp_flash3d1": "mpflash.ops",
          "asems": "asems.ops",
          "box3d_v2": "mpflash3D.ops",
          "box3d": "mpflash3D.ops",
          "box3d_slab": "mpflash3D.ops",
          "box3d_slab0": "mpflash3D.ops"}
        
        # find out which pulse sequence by peeking at the procpar
        pseq, flag = getPulseSeq(datadir)
        key = not flag and pseq or  pseq + reduce(lambda s1,s2: s1+s2, map(str, flag))
        if not oplistBySeq.has_key(key):
            raise RuntimeError(
              "No default operations found for pulse sequence '%s'"%key)
        opfilename = oplistBySeq[key]
        print "Using default oplist %s."%opfilename
        return recon.conf.getOplistFileName(opfilename)
        
    #-------------------------------------------------------------------------
    def configureOperations(self, opfileptr):
        """
        Creates an OrderedConfigParser object to parse the config file.
     
        Returns a list of (opclass, args) pairs by querying _opmanager for 
        the operation class by opname, and querying the OrderedConfigParser 
        for items (argumentss) by section (opname)
        
        @param opfileptr: fileptr of operations config file.
        @return: a list of operation pairs (operation, args).
        """
        config = OrderedConfigParser()
        config.readfp(opfileptr)
        opname = lambda k: k.rsplit(".",1)[0]
        return [
          (self._opmanager.getOperation(opname(opkey)), dict(config.items(opkey)))
          for opkey in config.sections()]

    #-------------------------------------------------------------------------
    def confirmOps(self, opseq):
        """This routine currently looks at the file i/o ops to make sure they
        are in a sane order. This routine might be expanded to double-check
        other sequence requirements
        """
        # make sure ReadImage is first op and only happens once,
        # if not change things around
        class_list = [opclass for (opclass, _) in opseq]
        read_op = self._opmanager.getOperation('ReadImage')
        op_spot = class_list.index(read_op)
        if op_spot > 0:
            (_, read_args) = opseq[op_spot]
            opseq.__delitem__(op_spot)
            opseq.insert(0,(read_op, read_args))
            # need to keep class_list synced for next step
            class_list.__delitem__(op_spot)
            class_list.insert(0,read_op)
        op_count = class_list.count(read_op)
        n = 1
        while op_count > 1:
            n += class_list[n:].index(read_op)
            opseq.__delitem__(n)
            class_list.__delitem__(n)
            op_count -= 1
        # print warning if WriteImage isn't the last op
        write_op = self._opmanager.getOperation('WriteImage')
        if class_list.index(write_op) != (len(class_list)-1):
            print "WARNING! Operation sequence doesn't end with "\
                  "WriteImage."

    #-------------------------------------------------------------------------
    def getOptions(self):
        """
        Bundle command-line arguments and options into a single options
        object, including a resolved list of callable data operations.
    
        Uses OptionParser to fill in the options list from command line input; 
        appends volume range specifications, and input/output directories as
        options; asks for an index of requested operations from
        configureOperations()
        """
    
        options, args = self.parse_args()
        options.vrange = parseVolRangeOption(options.vol_range, self)

        # Two usages are special cases:
        # oplist example will be handled after an oplist is found..
        # operation help doesn't require arguments, so handle it here
        if options.ophelp is not None:
            self._printOpHelp(options.ophelp)
            sys.exit(0)
        if options.opsexample:
            if not args:
                self.error("You must provide a fid-directory to get an example oplist")
            # just append a false output file name to be safe
            args.append("foo")
        
        # Recon can be run with these combos defined:
        # (_, _) (first logic stage, try to find fid files in pwd)
        # (args, _) (2nd logic stage, try to find default oplist)
        # (_, oplist) (3rd logic stage, try a couple things here)
        # (args,oplist) (last stage, not ambiguous)
        if not options.oplist and len(args) != 2:
            try:
                args = self._argsFromPWD()
            except:
                self.print_help()
                sys.exit(0)
        if not options.oplist:
            options.oplist = self._findOplist(args[0])
        options.operations = self.configureOperations(open(options.oplist,'r'))
        if len(args) != 2:
            # only care if we need to set up ReadImage and WriteImage later
            if not (self._running('ReadImage', options.operations) and \
                    self._running('WriteImage', options.operations)):
                args = self._argsFromPWD()
        if not self._running('ReadImage', options.operations):
            # append ReadImage op to BEGINNING of list
            op_args = {'filename': os.path.abspath(args[0]),
                       'vrange': options.vrange}
            opclass = self._opmanager.getOperation('ReadImage')
            options.operations.insert(0,(opclass, op_args))
        if not self._running('WriteImage', options.operations):
            op_args = {'filename': os.path.abspath(args[1]),
                       'suffix': options.suffix,
                       'filedim': options.filedim,
                       'format': options.file_format,
                       'datatype': options.output_datatype}
            opclass = self._opmanager.getOperation('WriteImage')
            options.operations.append((opclass, op_args))
        # run some checks on the operations sequence
        self.confirmOps(options.operations)

        if options.fastArray:
            global _FAST_ARRAY
            _FAST_ARRAY = True
        return options

    #-------------------------------------------------------------------------
    def runOperations(self, operations, image, runlogger):
        "Run the given list of operation objects and record into runlogger."
        for operation in operations:
            operation.log("Running")
            if operation.run(image) == -1:
                raise RuntimeError("critical operation failure")
                sys.exit(1)
            runlogger.logop(operation)

    #-------------------------------------------------------------------------
    def run(self):
        """
        Run the recon tool.
        
        Asks for options from self.getOptions(); starts RunLogger object;
        initializes FidImage object from the fid and procpar in data directory;
        loops through image operation battery; saves processed image
        """

        # Parse command-line options.
        options = self.getOptions()

        if options.opsexample:
            self._printOplistEx(options.oplist)
            sys.exit(0)

        runlogger = RunLogger(file(options.log_file,'w'))

        # Load k-space image from the fid file and log it.
        reader_class, reader_args = options.operations[0]
        reader = reader_class(**reader_args)
        image = reader.run()
        runlogger.logop(reader)        
        
        # Log some parameter info to the console.
        if hasattr(image, 'logParams'): image.logParams()

        # Instantiate the operations declared in oplist file.
        operations = [opclass(**args)
                      for opclass,args in options.operations[1:]]

        # Run the operations... don't catch exceptions, I want the traceback
        image.runOperations(operations, logger=runlogger)

        runlogger.setExecutable()
