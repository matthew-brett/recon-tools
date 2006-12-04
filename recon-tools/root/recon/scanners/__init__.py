from recon.imageio import ReconImage


class ScannerImage (ReconImage):
    """
    Interface for image data originating from an MRI scanner.
    This class of objects should be able to go through all of the operations,
    especially the artifact correction operations.

    In order to be used with certain Operations, this class guarantees the
    definition of certain system/scanning parameters.

    """

    necessary_params = dict((
        ('T_pe', 'time to scan from k(y=a,x=0) to k(y=a+1,x=0)'),
        ('phi', 'euler angle about z'),
        ('psi', 'euler angle about y'),
        ('theta', 'euler angle about x'),
        ('delT', 'sampling dwell time'),
        ('echo_time', 'time-to-echo'),
        ('asym_times', 'list of te times in an asems scan'),
        ('acq_order', 'order that the slices were acquired'),
        ('nseg', 'number of sampling segments'),
        ('sampstyle', 'style of sampling: linear, centric, interleaved')
    ))

    def __init__(self):
        # should set up orientation info
        self.check_attributes()


    def check_attributes(self):
        for key in ScannerImage.necessary_params.keys():
            if not hasattr(self, key):
                raise AttributeError("This is not a complete ScannerImage, "\
                               "missing parameter %s"%key)
            

    
    
