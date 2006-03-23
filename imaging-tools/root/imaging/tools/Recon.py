"Defines a command-line interface to the recon tool."
from optparse import OptionParser, Option
 
import imaging.conf
from imaging.operations.WriteImage import ANALYZE_FORMAT, NIFTI_SINGLE, \
    NIFTI_DUAL, MAGNITUDE_TYPE, COMPLEX_TYPE, WriteImage  
from imaging.tools import OrderedConfigParser, ConsoleTool
from imaging.operations import OperationManager, RunLogger, WriteImage
from imaging.varian.FidImage import getPulseSeq

##############################################################################
class Recon (ConsoleTool):
    """
    Handle command-line aspects of the recon tool.
    @cvar options: tuple of Option objs, filled in by OptionParser
    """

    _opmanager = OperationManager()
    output_format_choices = ( 
      ANALYZE_FORMAT, 
      NIFTI_DUAL, 
      NIFTI_SINGLE) 
    output_datatype_choices= (MAGNITUDE_TYPE, COMPLEX_TYPE) 
    default_logfile = "recon.log"

    options = (

      Option("-c", "--config", dest="config", type="string",
        default=None, action="store",
        help="Name of the config file describing operations and operation" \
        " parameters."),

      Option("-r", "--vol-range", dest="vol_range", type="string",
        default=":", action="store",
        help="Which image volumes to reconstruct.  Format is start:end, "
        "where either start or end may be omitted, indicating to start "\
        "with the first or end with the last respectively.  The index of "\
        "the first volume is 0.  The default value is a single colon "\
        "with no start or end specified, meaning process all image volumes.  "\
        "(NOTE: this option refers specifically to *image* volumes, not to "\
        "reference scans, so that the first image volume means the first "\
        "found after any reference scans.)"),

      Option("-f", "--file-format", dest="file_format", action="store",
        type="choice", default=WriteImage.ANALYZE_FORMAT,
        choices=output_format_choices,
        help="""{%s}
        analyze: Save individual image for each frame in analyze format.
        nifti dual: save nifti file in (hdr, img) pair.
        nifti single: save nifti file in single-file format."""%\
          ("|".join(output_format_choices))),

      Option("-y", "--output-data-type", dest="output_datatype",
        type="choice", default=MAGNITUDE_TYPE, action="store",
        choices=output_datatype_choices,
        help="""{%s}
        Specifies whether output images should contain only magnitude or
        both the real and imaginary components (only valid for analyze
        format)."""%("|".join(output_datatype_choices))),

      Option("-l", "--log-file", default=default_logfile,
        help="where to record reconstruction details ('%s' by default)"\
             %default_logfile))

    #-------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self.set_usage("usage: %prog [options] oplist datadir")
        self.add_options(self.options)

    #-------------------------------------------------------------------------
    def configureOperations(self, configfile):
        """
        Creates an OrderedConfigParser object to parse the config file.
     
        Returns a list of (opclass, args) pairs by querying _opmanager for 
        the operation class by opname, and querying the OrderedConfigParser 
        for items (argumentss) by section (opname)
        
        @param configfile: filename of operations config file.
        @return: a list of operation pairs (operation, args).
        """
        config = OrderedConfigParser()
        config.read(configfile)
        return [
          (self._opmanager.getOperation(opname), dict(config.items(opname)))
          for opname in config.sections()]

    #-------------------------------------------------------------------------
    def parseVolRange(self, vol_range):
        """
        Separates out the command-line option volume range into distinct numbers
        @param vol_range: volume range as x:y
        @return: vol_start = x, vol_end = y
        """
        parts = vol_range.split(":")
        if len(parts) < 2: self.error(
          "The specification of vol-range must contain a colon separating "\
          "the start index from the end index.")
        try: vol_start = int(parts[0] or 0)
        except ValueError: self.error(
          "Bad vol-range start index '%s'.  Must be an integer."%parts[0])
        try: vol_end = int(parts[1] or -1)
        except ValueError: self.error(
          "Bad vol-range end index '%s'. Must be an integer."%parts[1])
        return vol_start, vol_end

    #-------------------------------------------------------------------------
    def findConfig(self, datadir):
        "Determine which stock oplist to use based on pulse sequence."
        
        oplistBySeq = {
          "epidw": "epi.ops",
          "epi":   "epi.ops",
          "gems":  "gems.ops",
          "mp_flash3d0": "mpflash3D.ops",
          "mp_flash3d1": "mpflash.ops",
          "asems": "asems.ops" }
        
        # find out which pulse sequence by peeking at the procpar
        pseq, flag = getPulseSeq(datadir)
        key = not flag and pseq or  pseq + reduce(lambda s1,s2: s1+s2, map(str, flag))
        if not oplistBySeq.has_key(key):
            raise RuntimeError(
              "No default operations found for pulse sequence '%s'"%key)
        opfilename = oplistBySeq[key]
        print "Using default oplist %s."%opfilename
        return imaging.conf.getConfigFileName(opfilename)
        
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
        if len(args) != 2:
            self.error("Expecting 2 arguments: datadir ouput")

        # treat the raw args as named options
        options.datadir, options.outfile = args

        # use stock oplist if none specified
        if not options.config: options.config = self.findConfig(options.datadir)

        # parse vol-range
        options.vol_start, options.vol_end = \
          self.parseVolRange(options.vol_range)

        # configure operations
        options.operations = self.configureOperations(options.config)

        return options

    #-------------------------------------------------------------------------
    def runOperations(self, operations, image, runlogger):
        "Run the given list of operation objects and record into runlogger."
        for operation in operations:
            operation.log("Running")
            operation.run(image)
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

        runlogger = RunLogger(file(options.log_file,'w'))

        # Load k-space image from the fid file.
        image = self._opmanager.getOperation("ReadImage")(
          filename=options.datadir, format="fid").run()

        # Log some parameter info to the console.
        image.logParams()

        # Instantiate the operations declared in config file.
        operations = [opclass(**args) for opclass,args in options.operations]

        # Add an operation for saving data.
        operations.append(
           WriteImage.WriteImage(
            filename=options.outfile,
            format=options.file_format,
            datatype=options.output_datatype))

        # Run the operations.
        self.runOperations(operations, image, runlogger)
