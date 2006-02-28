"Defines a command-line interface to Recon"
from optparse import OptionParser, Option

from imaging.tools import OrderedConfigParser
from imaging.varian.FidImage import FIDL_FORMAT, VOXBO_FORMAT, ANALYZE_FORMAT,\
  NIFTI_SINGLE, NIFTI_DUAL, MAGNITUDE_TYPE, COMPLEX_TYPE, FidImage
from imaging.operations import OperationManager, RunLogger


##############################################################################
class Recon (OptionParser):
    """
    Handle command-line aspects of the recon tool.
    @cvar options: tuple of Option objs, filled in by OptionParser
    @cvar _opmanager: OperationManager used to find out opclasses given opnames
    """

    _opmanager = OperationManager()
    output_format_choices = (FIDL_FORMAT, VOXBO_FORMAT, ANALYZE_FORMAT, NIFTI_DUAL, NIFTI_SINGLE)
    output_datatype_choices= (MAGNITUDE_TYPE, COMPLEX_TYPE)
    options = (

          Option("-c", "--config", dest="config", type="string",
            default="recon.cfg", action="store",
            help="Name of the config file describing operations and operation"\
            " parameters."),

          Option("-r", "--vol-range", dest="vol_range", type="string", default=":",
            action="store",
            help="Which image volumes to reconstruct.  Format is start:end, where "\
            "either start or end may be omitted, indicating to start with the "\
            "first or end with the last respectively.  The index of the first "\
            "volume is 0.  The default value is a single colon with no start "\
            "or end specified, meaning process all volumes.  (Note, this option "\
            "refers specifically to image volumes, not to reference scans.)"),

          Option("-f", "--file-format", dest="file_format", action="store",
            type="choice", default=ANALYZE_FORMAT, choices=output_format_choices,
            help="""{%s}
            fidl: save floating point file with interfile and 4D analyze headers.
            analyze: Save individual image for each frame in analyze format.
            voxbo: Save in tes format.
            nifti dual: save nifti file in (hdr, img) pair.
            nifti single: save nifti file in single-file format."""%("|".join(output_format_choices))),

          Option("-t", "--tr", dest="TR", type="float", action="store",
            help="Use the TR given here rather than the one in the procpar."),

          Option("-y", "--output-data-type", dest="output_data_type",
            type="choice", default=MAGNITUDE_TYPE, action="store",
            choices=output_datatype_choices,
            help="""{%s}
            Specifies whether output images should contain only magnitude or
            both the real and imaginary components (only valid for analyze
            format)."""%("|".join(output_datatype_choices))),

          Option("-l", "--log-file", default="recon.log",
            help="where to record reconstruction details ('recon.log' by default)")
        )

    #-------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self.set_usage("usage: %prog [options] data output")
        self.add_options(self.options)

    #-------------------------------------------------------------------------
    def configureOperations(self, configfile):
        """
        Creates an OrderedConfigParser object to parse the config file
     
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
    def getOptions(self):
        """
        Bundle command-line arguments and options into a single options
        object, including a resolved list of callable data operations.
    
        Uses OptionParser to fill in the options list from command line input; 
        appends volume range specifications, and input/output directories as options;
        asks for an index of requested operations from configureOperations()
        """
    
        options, args = self.parse_args()
        if len(args) != 2: self.error("Expecting 2 arguments: datadir ouput")

        # treat the raw args as named options
        options.datadir, options.outfile = args

        # parse vol-range
        options.vol_start, options.vol_end = \
          self.parseVolRange(options.vol_range)

        # configure operations
        options.operations = self.configureOperations(options.config)

        return options

    #-------------------------------------------------------------------------
    def run(self):
        """
        Run the recon tool.
        
        Asks for options from self.getOptions(); starts RunLogger object;
        initializes FidImage object from the fid and procpar in data directory;
        loops through image operation battery; saves processed image
        """

        # Get the filename names and options from the command line.
        options = self.getOptions()

        runlogger = RunLogger(file("recon.log",'w')) # file(options.log_file, 'w')

        # Load k-space image from the fid file.
        image = FidImage(options.datadir, options.TR)

        # Log some parameter info to the console.
        image.logParams()

        # Apply operations to the data
        for operation_class, args in options.operations:
            operation = operation_class(**args)
            print "running %s"%operation_class
            operation.run(image)
            runlogger.logop(operation)

        # Save data to disk.
        image.save(options.outfile, options.file_format, options.output_data_type)

