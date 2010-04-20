import numpy as np
from recon.operations import Operation, Parameter, ChannelAwareOperation
from recon.scanners import siemens_utils as sm_utils
from recon import util

class ReadoutResampling(Operation):
    """ This operation performs all resampling in the read-out direction.
    These resampling operations are a combination of ramp-sampling correction
    and positive/negative echo timing mis-matches.

    It does not YET reduce oversampling.

    It runs fairly slowly.
    """

    params=(
      Parameter(name="fov_lim", type="tuple", default=None),
      Parameter(name="mask_noise", type="bool", default=True),
      )
    
    @ChannelAwareOperation
    def run(self, image):
        Tr = image.T_ramp
        Tf = image.T_flat
        T0 = image.T0
        N1 = image.N1
        Tg = 2*Tr + Tf
        dt = (Tg - 2*T0)/(N1-1)
        As = Tf + Tr - T0**2/Tr
        t_axis = np.arange(N1) * dt + T0
        nr = image.n_ramp
        nf = image.n_flat
        pe_rev = image.pe_reflected
        pe_sampling = image.pe_sampling
        ref_rev = image.ref_reflected
        acs_sampling = image.acs_sampling
        acs_rev = image.acs_reflected
        acs_ref_rev = image.acs_ref_reflected

        # tn_regrid and neg_tn_regrid are the destination time points
        # for positive and negative lobe readouts
        tn_regrid = image.t_n1()
        neg_tn_regrid = Tg - tn_regrid
##         print t_axis
##         print tn_regrid, neg_tn_regrid
        
##         tn_regrid = np.empty((N1,), 'd')
##         neg_tn_regrid = np.empty_like(tn_regrid)
##         idx = np.arange(N1)
##         r_up = idx < (nr+1)
##         tn_regrid[r_up] = ( T0**2 + 2*idx[r_up]*Tr*As/(N1-1) )**0.5
##         r_fl = (idx >= (nr+1)) & (idx < (nf+1))
##         tn_regrid[r_fl] = idx[r_fl]*As/(N1-1) + Tr/2 + T0**2/(2*Tr)
##         r_dn = idx >= (nf+1)
##         tn_regrid[r_dn] = Tg - ( 2*Tr*(Tf+Tr-T0**2/(2*Tr) - idx[r_dn]*As/(N1-1)) )**.5
##         neg_tn_regrid = Tg - tn_regrid

        
        same_grad_diff = pe_rev[1] - pe_rev[0]
        neg_slicing = slice(pe_rev[0], pe_rev[-1]+1, same_grad_diff)
        pos_sampling = pe_sampling[:]
        for pe in pe_rev:
            pos_sampling.remove(pe)
        pos_slicing = slice(pos_sampling[0], pos_sampling[-1]+1, same_grad_diff)

        for c in xrange(image.n_chan):
            for v in xrange(image.n_vol):
                for s in xrange(image.n_slice):
                    r = image.cref_data[c,v,s].copy()
                    # polarity is -1 if the first ref was on a neg gradient
                    polarity = ref_rev[0]==0 and -1 or +1
                    m = sm_utils.simple_unbal_phase_ramp(r, nr, nf, polarity,
                                                         self.fov_lim,
                                                         self.mask_noise)
                    Tau = m * dt * N1 / (2*np.pi)
                    # these are the time points of the acquired sampled
                    tn = (t_axis - Tau)
##                     print Tau, '('+str(dt)+')'
                    # compute these in transpose 
                    op_pos = np.sinc( (tn_regrid[None,:] - tn[:,None]) / dt)
                    op_neg = np.sinc( (neg_tn_regrid[None,:] - tn[:,None]) / dt)
                    neg_block = image.cdata[c,v,s,neg_slicing].copy()
                    pos_block = image.cdata[c,v,s,pos_slicing].copy()
                    # these are L x N1, to be transformed by N1 x N1 operators
                    # such that S <-- [(N1 x N1)*(L x N1)^T]^T = (L x N1)*(N1 x N1)^T
                    image.cdata[c,v,s,neg_slicing] = np.dot(neg_block, op_neg)
                    image.cdata[c,v,s,pos_slicing] = np.dot(pos_block, op_pos)

        if not acs_sampling:
            return

        acs_offset = -acs_sampling[0]
        # refs in the EPI section are never interleaved
        n_refs = image.n_refs
        n_acs_seg = image.cacs_ref_data.shape[-2]/n_refs
        g = 0
        # transform the acs indicators to be segment-wise
        l = len(acs_sampling)
        acs_sampling = np.array(acs_sampling).reshape(n_acs_seg,
                                                      l/n_acs_seg)
        l = len(acs_rev)
        acs_rev = np.array(acs_rev).reshape(n_acs_seg, l/n_acs_seg)
        l = len(acs_ref_rev)
        acs_ref_rev = np.array(acs_ref_rev).reshape(n_acs_seg, l/n_acs_seg)
        if len(image.cacs_data.shape) < 5:
            # pad with a rep dimension
            ashape = list(image.cacs_data.shape)
            rshape = list(image.cacs_ref_data.shape)
            ashape.insert(1,1)
            rshape.insert(1,1)
            image.cacs_data.shape = tuple(ashape)
            image.cacs_ref_data.shape = tuple(rshape)
            
        n_acs_rep = image.cacs_data.shape[1]
        for c in xrange(image.n_chan):
            for r in xrange(n_acs_rep):
                for s in xrange(image.n_slice):
                    for g in xrange(n_acs_seg):

                        acs_rev_g = acs_rev[g]
                        sampling_g = acs_sampling[g].tolist()
                        same_grad_diff = acs_rev_g[1] - acs_rev_g[0]
                        for acs in acs_rev_g:
                            sampling_g.remove(acs)
                        neg_slicing = slice(acs_rev_g[0]+acs_offset,
                                            acs_rev_g[-1]+acs_offset,
                                            same_grad_diff)
                        pos_slicing = slice(sampling_g[0]+acs_offset,
                                            sampling_g[-1]+acs_offset,
                                            same_grad_diff)


                        rd = image.cacs_ref_data[c,r,s,g*n_refs:(g+1)*n_refs].copy()
                        polarity = acs_ref_rev[g,0]==0 and -1 or +1
                        m = sm_utils.simple_unbal_phase_ramp(rd, nr, nf,
                                                             polarity,
                                                             self.fov_lim,
                                                             self.mask_noise)
                        Tau = m * dt * N1 / (2*np.pi)
    ##                     print Tau, '('+str(dt)+')'
                        # these are the time points of the acquired sampled
                        tn = t_axis - Tau
                        # compute these in transpose 
                        op_pos = np.sinc((tn_regrid[None,:]-tn[:,None])/dt)
                        op_neg = np.sinc((neg_tn_regrid[None,:]-tn[:,None])/dt)

                        neg_block = image.cacs_data[c,r,s,neg_slicing].copy()
                        pos_block = image.cacs_data[c,r,s,pos_slicing].copy()

                        # these are L x N1, to be multiplied by N1 x N1
                        # resampling operators such that
                        # S <-- [(N1 x N1)*(L x N1)^T]^T = (L x N1)*(N1 x N1)^T
                        image.cacs_data[c,r,s,neg_slicing] = np.dot(neg_block,
                                                                    op_neg)
                        image.cacs_data[c,r,s,pos_slicing] = np.dot(pos_block,
                                                                    op_pos)


        image.use_membuffer(0)
