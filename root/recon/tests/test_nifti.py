import os
import glob
from numpy.testing import *
import numpy as np
from recon import nifti
from recon import imageio
from recon.util import integer_ranges

@dec.setastest(False)
def dtypes_test(image, iomodule, output_type):
    """
    This will generically test analyze and nifti image i/o..
    *image is some ReconImage with random or real data.
    *iomodule is either recon.nifti or recon.analyze
    *output_type is the string identifier for the output (eg 'analyze')
    """

    # The best you can hope for with float->integer conversion is that
    # your dynamic range is preserved to the best extent possible..
    # so this test will see if the min and max values are preserved,
    # and that the log2 of the max-min is more or less equal to log2
    # of the max value of the integer type

    # this "should" be 1.0, but maybe it works in general??
    imscale = image.scaling
    image_max = np.abs(image[:]).max()
    image_min = np.abs(image[:]).min()
    drange_ratio = (image_max-image_min)/image_max
    for datatype, dt in imageio.recon_output2dtype.items():
        if dt not in iomodule.dtype2datatype.keys():
            continue
        image.writeImage('out', format_type=output_type,
                         datatype=datatype)
        image2 = iomodule.readImage('out')
        imscale2 = image2.scaling
        msg = "failed at dt=%s with scale=%f"%(dt,imscale2)
        im_max2 = np.abs(image2[:]).max()
        im_min2 = np.abs(image2[:]).min()
        # the significant digits that the original min value is
        # equal to the computed min value should be about the
        # order of magnitude quantization error + np.log10(image_min)
        sigdig = -np.log10(imscale2) + np.log10(image_min)
        
        assert_approx_equal(im_max2*imscale2, image_max, err_msg=msg)
        assert_approx_equal(im_min2*imscale2, image_min,
                            significant=sigdig, err_msg=msg)
        
        # let's say the relative dynamic range of the new image should
        # be about equal to the original
        drange_ratio2 = (im_max2-im_min2)/float(im_max2)
        if drange_ratio2 < 1.0:
            assert_approx_equal(drange_ratio2, drange_ratio,
                                significant=4, err_msg=msg)
        # else the drange is 100% maximized
    litter = glob.glob("out*")
    for f in litter:
        os.remove(f)
    


class test_nifti(TestCase):

    def setUp(self):
        self.pwd = os.path.split(__file__)[0]
        fname = os.path.join(self.pwd, 'filtered_func_data')
        self.image = nifti.readImage(fname, vrange=(0,1))

    def test_datatypes(self):
        # make a random valued complex image to cast into various dtypes
        image = self.image._subimage(1949.*np.random.rand(*self.image.shape) +
                                     1.j*1352*np.random.rand(*self.image.shape))
        dtypes_test(image, nifti, 'nifti-single')

    def test_orientation(self):
        rad_xform = np.array([[-1., 0., 0.],
                              [ 0., 1., 0.],
                              [ 0., 0., 1.],])

        neur_xform = np.array([[ 1., 0., 0.],
                               [ 0., 1., 0.],
                               [ 0., 0., 1.],])

        sag_xform = np.array([[ 0., 0.,-1.],
                              [ 1., 0., 0.],
                              [ 0., 1., 0.],])
        
        nif = nifti.readImage(os.path.join(self.pwd, 'avg152T1_LR_nifti'))
        assert_array_equal(nif.orientation_xform.tomatrix(), rad_xform)
        nif = nifti.readImage(os.path.join(self.pwd, 'avg152T1_RL_nifti'))
        assert_array_equal(nif.orientation_xform.tomatrix(), neur_xform)
        nif = nifti.readImage(os.path.join(self.pwd, 'gems2'))
        assert_array_equal(nif.orientation_xform.tomatrix(), sag_xform)
                              
        
if __name__=='__main__':
    NumpyTest(__file__.rstrip(".py")).test()
