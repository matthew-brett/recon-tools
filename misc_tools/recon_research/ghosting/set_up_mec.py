
def setup(mec, fpath, gpath):
    mec.push(dict(fpath=fpath, gpath=gpath))
    mec.execute('from recon import imageio, util')
    mec.execute('from recon.operations.ReorderSlices import ReorderSlices')
    mec.execute('from ghosting import search_driver_full_standalone as sdf')
    mec.execute('epi = imageio.readImage(fpath, vrange=(0,0), N1=128, do_rev=False, use_mmap=False)')
    mec.execute('ReorderSlices().run(epi)')
    mec.execute('import numpy as np')
    mec.execute("gmask = np.fromstring(open(gpath).read(), 'd')")
    mec.execute('gmask.shape = (22,64,128)')
    mec.execute('grad = util.grad_from_epi(epi)')
    mec.execute('from ghosting import search')
    mec.execute('from ghosting import search_driver_full_standalone as sdf')
    mec.execute('func = sdf.evalND_full_wrap')
    mec.execute('axes = range(4)')
    mec.execute('coefs = np.zeros(4)')
    mec.execute('args=(axes, epi, grad, coefs, 0, 0, 0, 1.0, None, gmask, None)')
    
