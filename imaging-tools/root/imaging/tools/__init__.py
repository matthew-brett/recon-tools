from ConfigParser import SafeConfigParser

##############################################################################
class OrderedConfigParser (SafeConfigParser):
    """
    Config parser which keeps track of the order in which sections appear.
    
    This class extends ConfigParser just slightly, giving it an ordered dictionary 
    to store sections, so that operations listed in the config file are performed 
    in the correct order. Otherwise, the functionality is that of ConfigParser.
    
    @cvar _sections: an odict with entries for given sections (opnames) and their related items (params)
    """

    #-------------------------------------------------------------------------
    def __init__(self, defaults=None):
        SafeConfigParser.__init__(self, defaults=defaults)
        import odict
        self._sections = odict.odict()


