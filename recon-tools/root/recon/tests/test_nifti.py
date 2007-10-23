import os
import glob
from numpy.testing import *
import numpy as N
set_package_path()
from recon import nifti
from recon import imageio
from recon.util import integer_ranges
restore_path()

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
    image_max = image[:].max()
    image_min = image[:].min()
    drange_ratio = (image_max-image_min)/image_max
    for datatype, dt in imageio.recon_output2dtype.items():
        if dt not in iomodule.dtype2datatype.keys():
            continue
        image.writeImage('out', format_type=output_type,
                         datatype=datatype)
        image2 = iomodule.readImage('out')
        imscale2 = image2.scaling
        msg = "failed at dt=%s"%dt
        #N.testing.assert_almost_equal(image2[:]*imscale2,
        #                              image[:]*imscale,decimal=-1,
        #                              err_msg=msg)
        im_max2 = image2[:].max()
        im_min2 = image2[:].min()
        # the significant digits that the original min value is
        # equal to the computed min value should be about the
        # order of magnitude quantization error + N.log10(image_min)
        sigdig = -N.log10(imscale2) + N.log10(image_min)
        assert_approx_equal(im_max2*imscale2, image_max, err_msg=msg)
        assert_approx_equal(im_min2*imscale2, image_min,
                            significant=sigdig, err_msg=msg)
        
        #let's say the relative dynamic range of the new image should
        #be about equal to the original
        drange_ratio2 = (im_max2-im_min2)/float(im_max2)
        if drange_ratio2 < 1.0:
            assert_approx_equal(drange_ratio2, drange_ratio,
                                significant=5)
        # else the drange is 100% maximized
    litter = glob.glob("out*")
    for f in litter:
        os.remove(f)
    


class test_nifti(NumpyTestCase):

    def setUp(self):
        self.pwd = os.path.split(__file__)[0]
        fname = os.path.join(self.pwd, 'filtered_func_data')
        self.image = nifti.readImage(fname, vrange=(0,1))

    def test_datatypes(self):
        image = self.image._subimage(2140.*N.random.rand(*self.image.shape).astype(N.float32))
        dtypes_test(image, nifti, 'nifti-single')

    def test_orientation(self):
        rad_xform = N.array([[-1., 0., 0.],
                             [ 0., 1., 0.],
                             [ 0., 0., 1.],])
        neur_xform = N.identity(3)
        sag_xform = N.array([[ 0., 0.,-1.],
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
