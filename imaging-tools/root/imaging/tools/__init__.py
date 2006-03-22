from ConfigParser import SafeConfigParser
from odict import odict

tool_map = odict((
  ("dumpheader", "imaging.tools.DumpHeader"),
  ("fdf2img", "imaging.tools.Fdf2Img"),
  ("getparam", "imaging.tools.GetParam"),
  ("imaging-doc", "imaging.tools.ImagingDoc"),
  ("recon", "imaging.tools.Recon"),
  ("viewimage", "imaging.tools.ViewImage")))


##############################################################################
class OrderedConfigParser (SafeConfigParser):
    """
    Config parser which keeps track of the order in which sections appear.
    
    This class extends ConfigParser just slightly, giving it an ordered
    dictionary to store sections, so that operations listed in the config file
    are performed in the correct order.  Otherwise, the functionality is that
    of ConfigParser.
    """

    #-------------------------------------------------------------------------
    def __init__(self, defaults=None):
        SafeConfigParser.__init__(self, defaults=defaults)
        self._sections = odict()


