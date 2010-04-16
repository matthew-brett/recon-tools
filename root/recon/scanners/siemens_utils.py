from __future__ import division

from recon.odict import odict
import struct
import cStringIO
import os
import numpy as np
from recon import util

class SiemensHeaderException (Exception):
    pass

class PagedMemmap(object):
    # this class will have linear, paged memmap access across a large file.
    # it can provide:
    # a generator to access item-by-item (with hooks to stride access)
    # a slicing mechanism that returns data up to the maximum page size

    # READ-ONLY ACCESS IS ENFORCED, 1D SHAPE IS ENFORCED
    def __init__(self, fpath, dtype, shape, **kwargs):
        kwargs['mode'] = 'r'
        new_shape = (np.prod(shape),)
        kwargs['shape'] = new_shape

        max_pg_size = long(2**31-1)

        memmap_bytes = long(new_shape[0]) * long(dtype.itemsize)

        mmb = memmap_bytes
        full_page_size = (max_pg_size // dtype.itemsize) * dtype.itemsize
        self.page_sizes = []
        while mmb > 0:
            self.page_sizes.append( min(mmb, full_page_size) )
            mmb -= full_page_size
        
        assert np.sum(self.page_sizes)==memmap_bytes, 'wrong page sizes'
        self._create_pages(fpath, dtype=dtype, **kwargs)
        
    def _create_pages(self, *args, **kwargs):
        offset = kwargs.get('offset', 0)
        self._mmaps = []
        bytes_per_item = kwargs['dtype'].itemsize
        for n in xrange(len(self.page_sizes)):
            kwargs['offset'] = offset
            kwargs['shape'] = (self.page_sizes[n]/bytes_per_item,)
            self._mmaps.append( np.memmap(*args, **kwargs) )
            offset += self.page_sizes[n]
        
        

def header_length(fname):
    return struct.unpack('<i', open(fname, 'rb').read(4))[0]

def header_string(fname):
    hdrlen = header_length(fname)
    if hdrlen > os.stat(fname).st_size:
        raise SiemensHeaderException('not a valid siemens .dat file')
    return open(fname, 'rb').read(hdrlen+4)

def strip_ascconv(hdr_str):

    # scan forward to the 2nd 'ASCCONV BEGIN' string
    pos = hdr_str.find('ASCCONV BEGIN')
##     hdr_str = hdr_str[pos+100:] # make sure to clear the 'ASCCONV' stuff
##     pos = hdr_str.find('ASCCONV BEGIN')
    hdr_str = hdr_str[pos:]
    # get to the first line of the ASCCONV section
    pos = hdr_str.find('\n')
    hdr_str = hdr_str[pos+1:]
    f = cStringIO.StringIO(hdr_str)
    field_dict = odict()
    for line in f.readlines():
        if line.find('ASCCONV END') >= 0:
            break
        line = line.strip('\n')
        field, val = line.split('=')
        field = field.strip()
        val = val.strip()
        val = val.split(' #')[0] # there may be comments like on this line??
        if(val.find('"') >= 0):
            field_dict[field] = val.strip('"')
        elif(val.find('0x') >= 0):
            field_dict[field] = int(val, 16)
        else:
            field_dict[field] = float(val)
    return field_dict
        
def condense_array(field_dict, field_base):
    field_keys = field_dict.keys()
    arrays_dict = odict()
    n = 0
    max_idx = len(field_keys)
    # advance to first array entry
    while(n < max_idx and field_keys[n].find(field_base) < 0):
        n += 1
    if n==max_idx:
        return arrays_dict
    # initialize all the arrays present.. do this by creating lists
    # for every new field where the index is 0
    idx = 0
    while(True):
        field = field_keys[n]
        if(field.find(field_base) < 0):
            break
        # field is like this: 'sSliceArray.asSlice[0].sPosition.dTra'
        # it gets decomposed into tail: '0].sPosition.dTra'
        tail = field.split('[')[-1]
        # then decomposed into the number and name: ['0', '.sPosition.dTra']
        idx_str, a_name = tail.split(']')
        idx = int(idx_str)
        if idx==0:
            arrays_dict[a_name[1:]] = [field_dict[field]]
        else:
            arrays_dict[a_name[1:]].append(field_dict[field])
        n += 1
    for a_name, a in arrays_dict.items():
        arrays_dict[a_name] = np.array(a)
    return arrays_dict

def condense_arrays(field_dict):
    array_fields = ('asCoilSelectMeas[0].aFFT_SCALE',
                    #'sSliceArray.anPos',
                    #'sSliceArray.anAsc',
                    'sSliceArray.asSlice',
                    )
    arrays_dict = odict()
    field_keys = field_dict.keys()
    max_idx = len(field_keys)
    for field_base in array_fields:
        n = 0
        arrays_dict[field_base] = odict()
        aset = arrays_dict[field_base]
        # advance to first array entry
        while(n < max_idx and field_keys[n].find(field_base) < 0):
            n += 1
        if n==max_idx:
            continue
        # initialize all the arrays present.. do this by creating lists
        # for every new field where the index is 0
        idx = 0
        while(True):
            field = field_keys[n]
            if(field.find(field_base) < 0):
                break
            # field is like this: 'sSliceArray.asSlice[0].sPosition.dTra'
            # it gets decomposed into tail: '0].sPosition.dTra'
            tail = field.split('[')[-1]
            # then decomposed into the number and name: ['0', '.sPosition.dTra']
            idx_str, a_name = tail.split(']')
            idx = int(idx_str)
            if idx==0:
                aset[a_name[1:]] = [field_dict[field]]
            else:
                aset[a_name[1:]].append(field_dict[field])
            n += 1
        for a_name, a in aset.items():
            aset[a_name] = np.array(a)
        
    return arrays_dict
        
def get_arrays(fname):
    hdrlen = header_length(fname)
    hdr_str = open(fname, 'rb').read(4+hdrlen)    
    d = strip_ascconv(hdr_str)
    adict = condense_arrays(d)
    return adict

def get_rotations(normal):
    x = np.array([1,0,0]); y = np.array([0,1,0]); z = np.array([0,0,1])
    yz_proj = np.dot(y, normal)*y + np.dot(z, normal)*z
    # <z, yz_proj> = vnorm(z)*vnorm(yz_proj)*cos(theta)
    theta = np.arccos( np.dot(z, yz_proj)/vnorm(yz_proj) )
    xz_proj = np.dot(x, normal)*x + np.dot(z, normal)*z
    psi = -np.arccos( np.dot(z, xz_proj)/vnorm(xz_proj) )
    xy_proj = np.dot(x, normal)*x + np.dot(y, normal)*y
    phi = np.arccos( np.dot(y, xy_proj)/vnorm(xy_proj) )
    return util.eulerRot(theta=theta, psi=psi, phi=phi)

## def get_quat(fname):
##     dat = MemmapDatFile(fname, nblocks=10)
##     mdh = MDH(dat[0]['hdr'])
##     q = Quaternion(i=mdh.quatI, j=mdh.quatJ, k=mdh.quatK)
##     # this is apparently (phase_dim, freq_dim, slice_dim) --> (x,y,z)
##     m = q.tomatrix()
##     m2 = np.eye(3, dtype='d')
##     m2[:,0] = m[:,1]; m2[:,1] = m[:,0]
##     m2[0:2] *= -1
##     return Quaternion(M=m2)


def vnorm(v):
    return np.dot(v,v)**0.5

import glob
def walk_func(arg, dirname, fnames):
    dats = glob.glob(dirname+"/*.dat")
    print "processing ", dirname
    for dat in dats:
        i = raw_input('process '+dat+'?  ')
        if i != 'n':
            print get_arrays(dat)
    
## from recon.scanners.siemens import MemmapDatFile, MDH
## from recon.util import Quaternion        
## def get_rotation_info(fname):
##     d = strip_ascconv(header_string(fname))
##     slice_arrays = condense_array(d, 'sSliceArray.asSlice')
##     in_plane_rot = slice_arrays.get('dInPlaneRot', [0])[0]
##     dat = MemmapDatFile(fname, nblocks=10)
##     mdh = MDH(dat[0]['hdr'])
##     quat = Quaternion(i=mdh.quatI, j=mdh.quatJ, k=mdh.quatK)
##     print "in_plane_rot:", in_plane_rot
##     print "quaternion:", quat.Q
##     print quat.tomatrix()
        
        
        
def mdh_quat_to_mat(mdh):
    w,x,y,z = mdh.quatW, mdh.quatI, mdh.quatJ, mdh.quatK
    xx = x*x; yy = y*y; zz = z*z; ww = w*w
    xy = x*y; xz = x*z; xw = x*w;
    yw = y*w; yz = y*z; zw = z*w;
    m = np.array([ [(ww+xx-yy-zz), 2*(xy-zw), 2*(xz+yw)],
                   [2*(xy+zw), (ww-xx+yy-zz), 2*(yz-xw)],
                   [2*(xz-yw), 2*(yz+xw), (ww-xx-yy+zz)] ])
    return m

def simple_unbal_phase_ramp(rdata, nramp, nflat, pref_polarity, 
                            fov_lim=None, mask_noise=True,
                            debug=False):
    N1 = rdata.shape[-1]
    rdata[:,:nramp+1] = 0.
    rdata[:,nflat+1:] = 0.
    irdata = util.ifft1(rdata, shift=True)

    if pref_polarity==1:
        # [(pos - neg), (neg - pos)]
        ref_funcs = irdata[:2] * irdata[1:3].conj()
    else:
        # [(pos - neg), (neg - pos)]        
        ref_funcs = irdata[:2].conj()*irdata[1:3]
    ref = ref_funcs[0]*ref_funcs[1].conjugate()
    ref_funcs_phs = np.angle(ref_funcs)
    pos_neg_diff = ref_funcs_phs[0]
    neg_pos_diff = ref_funcs_phs[1]
    
    ref_peak = np.abs(ref).max()

    # if an FOV limit is requested, then make sure the rest of the algorithm
    # only considers data within the limit
    if fov_lim:
        fov_max, fov = fov_lim
        dx = fov/N1
        x_grid = np.arange(-N1/2, N1/2) * dx
        fov_q1_mask = np.where(np.abs(x_grid)>fov_max,0,1)
    else:
        fov_q1_mask = np.ones((N1,), 'i')

    ref_peak = np.abs(ref*fov_q1_mask).max()
    if mask_noise:
        thresh = 0.1
        nz_q1_mask = np.where(np.abs(ref) > thresh*ref_peak, 1, 0)
        while nz_q1_mask.sum() < 10:
            thresh *= 0.5
            print 'relaxing threshold to include more points in the fit:', thresh
            nz_q1_mask = np.where(np.abs(ref) > thresh*ref_peak, 1, 0)
    else:
        nz_q1_mask = np.ones((N1,), 'i')

    ref_phs = (pos_neg_diff-neg_pos_diff)

    q1_mask = nz_q1_mask & fov_q1_mask
    # only do the unwrapping in the method if noisy patches have been rejected
    m = util.find_ramp(ref_phs/4., mask=q1_mask, do_unwrap=mask_noise,
                       debug=debug)
    
##     m,b,r = util.lin_regression(ref_phs/4, mask=nz_q1_mask & fov_q1_mask)
##     m = m[0]; b = b[0]
    return m
