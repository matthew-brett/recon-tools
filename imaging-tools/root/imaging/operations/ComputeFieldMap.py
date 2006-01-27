#import file_io
from Numeric import *
from LinearAlgebra import *
import math
from sliceview import sliceview
from imaging.util import exec_cmd, unwrap_phase
from imaging.imageio import write_analyze
from imaging.operations import Operation, Parameter


##############################################################################
class ComputeFieldMap (Operation):
    "Perform phase unwrapping and calculate field map"

    #-------------------------------------------------------------------------
    def run(self, image):
        if not hasattr(image._procpar, "asym_time"):
            self.log("No asym_time, can't compute field map.")
            return

        # Get procpar info.
        nvol = image.nvol
        #asym_time = image._procpar.asym_time
        #print "asym_time:",asym_time

        # Create image filenames.
        trange = range(nvol)
        wrapped = ["wrapped_%04d.img"%t for t in trange]
        phs_unwrapped = ["phs_unwrapped_%04d.img"%t for t in trange]
        #fmap_files = ["fmap_%02d.img"%t for t in trange[:-1]]
        #pixshift_files = ["pixshift_%02d.img"%t for t in trange[:-1]]
        #fmap_file_fitted = "fmap_fitted.img"
        #phase_pair = "phase_pair.img"
        #phase_pair_hdr = "phase_pair.hdr"

        # Unwrap phases.
        unwrapped_volumes = [unwrap_phase(volume) for volume in image.data]
        sliceview(unwrapped_volumes[0])
 
        # Read header info and set for output.
        #hdr = file_io.read_header(wrapped[0])
        #xdim = hdr['xdim']
        #ydim = hdr['ydim']
        #zdim = hdr['zdim']
        #xsize = hdr['xsize']
        #ysize = hdr['ysize']
        #zsize = hdr['zsize']
        #datatype = 3 # Floating point

        #for idx in range(nvol-1):

            # Concatenate adjacent unwrapped phase maps into a single 2-frame file.
        #    exec_cmd("cat %s %s > %s"%(phs_unwrapped[idx],phs_unwrapped[idx+1],phase_pair))
        #    hdr = file_io.create_hdr(xdim,ydim,zdim,2,xsize,ysize,zsize,1.,0,0,0,'Float',32,1.,'analyze',phase_pair,0)
        #    file_io.write_analyze_header(phase_pair_hdr,hdr)

            # Compute B-maps for concatenated unwrapped phase maps.
        #    exec_cmd("fugue -p %s --asym=%f --savefmap=%s --dwell=1. --saveshift=%s"%\
        #      (phase_pair,asym_time,fmap_files[idx],pixshift_files[idx]))

        # Use a linear regression to fit phases, then use the the fit to estimate field map.
        #if nvol > 2:
        #    sumx = 0.
        #    sumxsq = 0.
        #    sumy = zeros((zdim,ydim,xdim)).astype(Float)
        #    sumxy = zeros((zdim,ydim,xdim)).astype(Float)
        #    phasem1 = zeros((zdim,ydim,xdim)).astype(Float)
        #    for t in range(nvol):
        #        phs_data = file_io.read_file(phs_unwrapped[t])
        #        phase = phs_data['image']
        #        if t > 0:
                    # Ensure that phase is within 2*pi of the last phase values (Acq protocol should not allow phase wraps.)
        #            diff = phase - phasem1
        #            mask = where(equal(diff,0.),0.,1.)
        #            N = sum(mask.flat)
        #            offset = int(sum(diff.flat)/(N*2.*math.pi))
        #            print t,N,offset
        #            if offset != 0.:
        #                phase = phase - 2.*math.pi*offset*mask
        #                file_io.write_analyze("phase.img",phs_data['header'],phase)
        #        sumx = sumx + t
        #        sumxsq = sumxsq + t**2
        #        sumy = sumy + phase[:,:,:]
        #        sumxy = sumxy + t*phase[:,:,:]
        #        phasem1[:,:,:] = phase
        #    fmap = (nvol*sumxy - sumx*sumy)/(asym_time*(nvol*sumxsq - sumx**2))
        #    fmap = median_filter(fmap,3)
        #    fmap_hdr = file_io.create_hdr(xdim,ydim,zdim,1,xsize,ysize,zsize,1.,0,0,0,'Float',32,1.,'analyze',fmap_file_fitted,0)
        #    file_io.write_analyze(fmap_file_fitted,fmap_hdr,fmap)

        # cleanup leftover files
        sp = " "
        exec_cmd("/bin/rm %s %s %s %s %s"%(
          sp.join(wrapped), sp.join(phs_unwrapped), phase_pair))

        #======== Move this dwell time stuff into ProcParImageMixin =========
        # Determine dwell time.
        #if image._procpar.has_key('dwell'):
            #dwell_time = image._procpar.dwell
        #else:
            #trise = image._procpar.trise
            #gro = image._procpar.gro
            #gmax = image._procpar.gmax
            #np = image._procpar.np
            #sw = image._procpar.sw
            # This one give values that agree with Varian's
            #dwell_time = (trise*gro/gmax) + np/(2.*sw)
