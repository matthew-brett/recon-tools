from ConfigParser import SafeConfigParser
from optparse import OptionParser
from odict import odict
from recon import import_from

tool_map = odict((
  ("dumpheader", "DumpHeader"),
  ("fdf2img", "Fdf2Img"),
  ("getparam", "GetParam"),
  ("recon-doc", "ImagingDoc"),
  ("recon", "Recon"),
  ("recon_gui", "ReconGUI"),
  ("viewimage", "ViewImageTool")))
tool_names = tool_map.keys()

default_logfile = "recon.log"
#-----------------------------------------------------------------------------
def getToolByName(toolname):
    if not tool_map.has_key(toolname):
        raise ValueError("No tool called '%s'."%toolname)
    classname = tool_map[toolname]
    modulename = "recon.tools.%s"%classname
    try:
        return import_from(modulename, classname)
    except ImportError:
        raise RuntimeError("Tool class %s.%s not found."%(modulename,classname))


def parseVolRangeOption(vol_range, parser):
    """
    Separates out the command-line option volume range into distinct numbers
    @param vol_range: volume range as x:y
    @return: vol_start = x, vol_end = y
    """
    parts = vol_range.split(":")
    if len(parts) < 2: parser.error(
        "The specification of vol-range must contain a colon separating "\
        "the start index from the end index.")
    try: vol_start = int(parts[0] or 0)
    except ValueError: parser.error(
        "Bad vol-range start index '%s'.  Must be an integer."%parts[0])
    try: vol_end = int(parts[1] or -1)
    except ValueError: parser.error(
        "Bad vol-range end index '%s'. Must be an integer."%parts[1])
    return (vol_start, vol_end) != (0,-1) and (vol_start, vol_end) or ()
    

##############################################################################
class MultiKeyDict (odict):
    def _add_id(self, key):
        samekeys = [k for k in self.keys() if k.rsplit(".",1)[0] == key]
        return key+".%s"%len(samekeys)
    def __setitem__(self, key, item):
        odict.__setitem__(self, self._add_id(key), item)


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
        self._sections = MultiKeyDict()


##############################################################################
class ConsoleTool (OptionParser):
    command_name = ""
    description = ""
