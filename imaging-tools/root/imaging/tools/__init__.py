from ConfigParser import SafeConfigParser

##############################################################################
class OrderedConfigParser (SafeConfigParser):
    "Config parser which keeps track of the order in which sections appear."

    #-------------------------------------------------------------------------
    def __init__(self, defaults=None):
        SafeConfigParser.__init__(self, defaults=defaults)
        import odict
        self._sections = odict.odict()


