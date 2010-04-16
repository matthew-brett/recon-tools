import numpy as np
from recon.operations import Operation, Parameter, verify_scanner_image, \
     ChannelIndependentOperation

def change_slices_inplace(data, acq_order):
    holdershape = (2,) + data[:,0,:,:].shape
    out_of_place = np.ones(data.shape[-3])
    holder = np.empty(holdershape, data.dtype)
    k = 0
    while out_of_place.sum():
        p = k
        if not out_of_place[k]:
            k += 1
            continue
        skipto = acq_order[p]
        pstart = skipto
        
        if skipto == p:
            out_of_place[p] = 0
            k += 1
            continue
        
        t_idx = 0
        holder[t_idx] = data[:,p,:,:].copy()
        while True:
            holder[t_idx^1] = data[:,skipto,:,:].copy()
            data[:,skipto,:,:] = holder[t_idx]
            out_of_place[skipto] = 0
            p = skipto
            skipto = acq_order[p]
            if skipto==pstart:
                break
            t_idx = t_idx ^ 1
        k += 1

##############################################################################
class ReorderSlices (Operation):
    """
    Reorder image slices from acquisition order to physical order.
    """
    @ChannelIndependentOperation
    def run(self, image):
        if not verify_scanner_image(self, image): return
        acq_order = image.acq_order
        data = image[:,:,:,:]
        change_slices_inplace(data, acq_order)
        
        if hasattr(image, 'ref_data'):
            data = image.ref_data[:,:,:,:]
            change_slices_inplace(data, acq_order)
        if hasattr(image, 'acs_data') and image.acs_data is not None:
            data = image.acs_data[:,:,:,:]
            change_slices_inplace(data, acq_order)
        if hasattr(image, 'acs_ref_data') and image.acs_ref_data is not None:
            data = image.acs_ref_data[:,:,:,:]
            change_slices_inplace(data, acq_order)


#### WEAVE BLITZ CODE, ~ 1SEC FASTER ON 30 SLICES

##         code = """
##         using namespace blitz;
##         int k, p, pstart, skipto, tmp_idx;
##         k = 0;
##         while( sum(out_of_place) ) {
##           p = k;
##           //std::cout << "starting cycle at point " << k << std::endl;
##           if( !out_of_place(k) ) {
##             //std::cout << k << " is not out of place, moving on" << std::endl;
##             k++;
##             continue;
##           }

##           // keep track of the start of this loop
##           skipto = acq_order(p);
##           pstart = skipto;

##           // if it's already in the right place, update state and move on
##           if(skipto==p) {
##             //std::cout << k << " is already in place, moving on" << std::endl;
##             out_of_place(p) = 0;
##             k++;
##             continue;
##           }
##           tmp_idx = 0;
##           holder(tmp_idx,Range::all(), Range::all(), Range::all()) = \
##           data(Range::all(), p, Range::all(), Range::all());
##           //std::cout << "would hold " << p << " in holder(" << tmp_idx << ")"<<std::endl;

##           while(1) {
##             // save previous data at skipto point
##             //std::cout<<"would hold " <<skipto<<" in holder("<<(tmp_idx^1)<<")"<<std::endl;
##             holder(tmp_idx ^ 1, Range::all(), Range::all(), Range::all()) = \
##             data(Range::all(), skipto, Range::all(), Range::all());

##             // update data at skipto point
##             //std::cout<<"would move holder("<<tmp_idx<<") to "<<skipto;
##             //std::cout<<"  data from "<<p<<" is in place, so holder("<<tmp_idx<<") is free" << std::endl;
##             data(Range::all(), skipto, Range::all(), Range::all()) = \
##             holder(tmp_idx, Range::all(), Range::all(), Range::all());

##             // update state of skipto
##             out_of_place(skipto) = 0;
##             // get ready to go to next point
##             p = skipto;
##             skipto = acq_order(p);
##             //std::cout<<"moving from "<<p<<" to "<<skipto<<std::endl;
##             // if already visited, break loop
##             if(skipto==pstart) {
##               //std::cout<<"already been there, ending cycle"<<std::endl;
##               break;
##             }
            
##             tmp_idx ^= 1;
##           }
##           k++;
##         }

##         """
