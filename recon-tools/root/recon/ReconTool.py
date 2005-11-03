from ConfigParser import SafeConfigParser
from optparse import OptionParser, Option
from recon.FidImage import FIDL_FORMAT, VOXBO_FORMAT, SPM_FORMAT,MAGNITUDE_TYPE, COMPLEX_TYPE
from recon.operations import OperationManager
from recon.FidImage import FidImage

output_format_choices = (FIDL_FORMAT, VOXBO_FORMAT, SPM_FORMAT)
output_datatype_choices= (MAGNITUDE_TYPE, COMPLEX_TYPE)


##############################################################################
class OrderedConfigParser (SafeConfigParser):
    "Config parser which keeps track of the order in which sections appear."

    #-------------------------------------------------------------------------
    def __init__(self, defaults=None):
        SafeConfigParser.__init__(self, defaults=defaults)
        import odict
        self._sections = odict.odict()


##############################################################################
class ReconTool (OptionParser):
    "Handle command-line aspects of the recon tool."

    #-------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self._opmanager = OperationManager()
        opnames = self._opmanager.getOperationNames()
        self.set_usage("usage: %prog [options] data output")
        self.add_options((
          Option( "-c", "--config", dest="config", type="string",
            default="recon.cfg", action="store",
            help="Name of the config file describing operations and operation"\
            " parameters."),
          Option( "-r", "--vol-range", dest="vol_range", type="string", default=":",
            action="store",
            help="Which volumes to reconstruct.  Format is start:end, where "\
            "either start or end may be omitted, indicating to start with the "\
            "first or end with the last respectively.  The index of the first "\
            "volume is 0.  The default value is a single colon with no start "\
            "or end specified, meaning process all volumes."),
          Option( "-n", "--nvol", dest="nvol_to_read", type="int", default=0,
            action="store",
            help="Number of volumes within run to reconstruct." ),
          Option( "-s", "--frames-to-skip", dest="skip", type="int", default=0,
            action="store",
            help="Number of frames to skip at beginning of run." ),
          Option( "-f", "--file-format", dest="file_format", action="store",
            type="choice", default=FIDL_FORMAT, choices=output_format_choices,
            help="""{%s}
            fidl: save floating point file with interfile and 4D analyze headers.
            spm: Save individual image for each frame in analyze format.
            voxbo: Save in tes format."""%("|".join(output_format_choices)) ),
          Option( "-p", "--phs-corr", dest="phs_corr", default="", action="store",
            help="Dan, please describe the action of this option..."),
          Option( "-a", "--save-first", dest="save_first", action="store_true",
            help="Save first frame in file named 'EPIs.cub'." ),
          Option( "-t", "--tr", dest="TR", type="float", action="store",
            help="Use the TR given here rather than the one in the procpar." ),
          Option( "-l", "--flip-left-right", dest="flip_left_right",
            action="store_true", help="Flip image about the vertical axis." ),
          Option( "-y", "--output-data-type", dest="output_data_type",
            type="choice", default=MAGNITUDE_TYPE, action="store",
            choices=output_datatype_choices,
            help="""{%s}
            Specifies whether output images should contain only magnitude or
            both the real and imaginary components (only valid for analyze
            format)."""%("|".join(output_datatype_choices)) )
        ))

    #-------------------------------------------------------------------------
    def configureOperations(self, configfile):
        """
        @return a list of operation pairs (operation, args).
        @param configfile: filename of operations config file.
        """
        config = OrderedConfigParser()
        config.read(configfile)
        return [
          (self._opmanager.getOperation(opname), dict(config.items(opname)))
          for opname in config.sections()]

    #-------------------------------------------------------------------------
    def parseVolRange(self, vol_range):
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
        """
        options, args = self.parse_args()
        if len(args) != 2: self.error(
          "Expecting 2 arguments: datadir ouput" )

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
        "Run the epi_recon tool."

        # Get the filename names and options from the command line.
        options = self.getOptions()

        # Load data from the fid file.
        data = FidImage(options.datadir, options)

        # Log some parameter info to the console.
        data.logParams()

        # Now apply the various data manipulation and artifact correction operations
        # to the time-domain (k-space) data which is stored in the arrays
        # data_matrix and nav_data as well as the ancillary data arrays ref_data and
        # ref_nav_data. The operations are applied by looping over the list of
        # operations specified in the config file. Each operation acts in an
        # independent manner upon the data arrays.
        for operation, args in options.operations:
            operation(**args).run(options, data)

        # Save data to disk.
        data.save(options.outfile, options.file_format, options.output_data_type)

