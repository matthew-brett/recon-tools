from optparse import OptionParser, Option
from recon.tools import ConsoleTool, default_logfile, parseVolRangeOption
from recon.imageio import readImage
from recon.visualization.recon_gui import recon_gui

class ReconGUI (ConsoleTool):
    "Launch the GUI front-end to Recon Tools"

    usage = 'usage: %prog [image]\n', __doc__

    options = (
        Option("-r", "--vol-range", dest="vol_range", type="string",
        default=":", action="store",
        help="Which image volumes to reconstruct. Format is start:end, "\
        "where either start or end may be omitted, indicating to start "\
        "with the first or end with the last respectively. The index of "\
        "the first volume is 0. The default value is a single colon "\
        "with no start or end specified, meaning process all image volumes."),

        Option("-l", "--log-file", default=default_logfile,
        help="where to record reconstruction details ('%s' by default)"\
        %default_logfile),

    )

    #-------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self.add_options(self.options)

    def run(self):
        opts, args = self.parse_args()
        if args:
            vrange = parseVolRangeOption(opts.vol_range, self)
            try:
                image = readImage(args[0], vrange=vrange)
            except:
                image = None
                print 'Could not find image %s, try loading manually'%args[0]
                
        else:
            image = None
        recon_gui(image=image, logfile=opts.log_file)
        
        
