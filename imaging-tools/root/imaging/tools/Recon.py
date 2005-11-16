from ConfigParser import SafeConfigParser
from optparse import OptionParser, Option
from imaging.varian.FidImage import (FIDL_FORMAT, VOXBO_FORMAT, ANALYZE_FORMAT,
  MAGNITUDE_TYPE, COMPLEX_TYPE, FidImage)
from imaging.operations import OperationManager

output_format_choices = (FIDL_FORMAT, VOXBO_FORMAT, ANALYZE_FORMAT)
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
class Recon (OptionParser):
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
            help="Which image volumes to reconstruct.  Format is start:end, where "\
            "either start or end may be omitted, indicating to start with the "\
            "first or end with the last respectively.  The index of the first "\
            "volume is 0.  The default value is a single colon with no start "\
            "or end specified, meaning process all volumes.  (Note, this option "\
            "refers specifically to image volumes, not to reference scans.)"),
          Option( "-f", "--file-format", dest="file_format", action="store",
            type="choice", default=ANALYZE_FORMAT, choices=output_format_choices,
            help="""{%s}
            fidl: save floating point file with interfile and 4D analyze headers.
            analyze: Save individual image for each frame in analyze format.
            voxbo: Save in tes format."""%("|".join(output_format_choices)) ),
          Option( "-t", "--tr", dest="TR", type="float", action="store",
            help="Use the TR given here rather than the one in the procpar." ),
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
        "Run the epi_recon tool."

        # Get the filename names and options from the command line.
        options = self.getOptions()

        # Load k-space image from the fid file.
        image = FidImage(options.datadir, options.TR)

        # Log some parameter info to the console.
        image.logParams()

        # Now apply the various data manipulation and artifact correction
        # operations to the time-domain (k-space) data which is stored in the
        # image attributes data and nav_data as well as the ancillary data arrays
        # ref_data and ref_nav_data. The operations are applied by looping over
        # the list of operations specified in the config file. Each operation
        # acts independently upon the data.
        for operation, args in options.operations: operation(**args).run(image)

        # Save data to disk.
        image.save(options.outfile, options.file_format, options.output_data_type)

