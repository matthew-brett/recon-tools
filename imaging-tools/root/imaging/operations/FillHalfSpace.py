from imaging.operations import Operation, Parameter
from pylab import zeros, Complex32, Float, arange, reshape, rot90, conjugate, angle, sum_flat, exp, Complex, NewAxis, cos, pi
from imaging.util import embedIm
from imaging.operations.ForwardFFT import ForwardFFT as FwdFFT
from imaging.operations.InverseFFT import InverseFFT as InvFFT

class FillHalfSpace (Operation):

    params=(
        Parameter(name="fill_size", type="int", default=0,
                  description="the new number of rows to fill to in k-space"),)

    def phaseMap(self, vol):
        (ns, ny, nx) = vol.data.shape
        ny = self.fill_size
        y0 = self.fill_size/2
        fill_vol = zeros((1, ns,ny,nx), Complex32)
        fill_vol[:,:,y0-self.over_fill:y0+self.over_fill,:] = \
                     vol.data[:,0:self.over_fill*2,:]
        fill_image = vol._subimage(fill_vol)
        self.IFFT2D.run(fill_image)
        #return fill_image
        phase_map = angle(fill_image.data[0])
        return phase_map

    def imageFromFill(self, vol):
        (ns, ny, nx) = vol.data.shape
        ny = self.fill_size
        fill_vol = zeros((1,ns,ny,nx), Complex32)
        for s in range(ns):
            embedIm(vol.data[s], fill_vol[0,s], self.fill_rows, 0)
        fill_image = vol._subimage(fill_vol)
        self.IFFT2D.run(fill_image)
        return fill_image.data[0]

    def HermitianFill(self, Im, n_fill_rows):
        np, nf = Im.shape
        x1 = np/2+1  # this is where x=1
        # catch x=0 with sub-matrix symmetric with A
        Acomp = Im[np-n_fill_rows+1:,x1-1:]
        Bcomp = Im[np-n_fill_rows+1:,1:x1-1]
        Im[1:n_fill_rows,1:x1] = conjugate(rot90(Acomp, k=2))
        Im[1:n_fill_rows,x1:] = conjugate(rot90(Bcomp, k=2))
        Im[0] = 0
        Im[0:n_fill_rows,0] = 0

    def cookImage(self, vol):
        theta = self.phaseMap(vol)
        filled = self.imageFromFill(vol)
        (ns, _, nx) = vol.data.shape
        cooked = zeros((1,ns,self.fill_size,nx), Complex)
        cooked[0] = filled*exp(-1.j*theta)
        cookedIm = vol._subimage(cooked)
        self.FFT2D.run(cookedIm)
        return cookedIm.data

    def restoreMeasured(self, filled, measured):
        filled[:,self.fill_rows:,:] = measured[:].astype(filled.typecode())

    def mergeFill(self, filled, measured, winsize=8):
        mergept = self.fill_rows
        fill_win = 0.5*(1 + cos(pi*(arange(winsize)/float(winsize))))
        measured_win = 0.5*(1 + cos(pi + pi*(arange(winsize)/float(winsize))))
        # put back original measured data in rows mergept+winsize and on
        filled[:,mergept+winsize:,:] = measured[:,winsize:,:]
        # merge measured data with filled data in winsize merge region
        filled[:,mergept:mergept+winsize,:] = \
               fill_win[:,NewAxis]*filled[:,mergept:mergept+winsize,:] + \
               measured_win[:,NewAxis]*measured[:,:winsize,:]

    def cookImage2(self, vol):
        theta = self.phaseMap(vol)
        mag = abs(self.imageFromFill(vol))
        (ns, _, nx) = vol.data.shape
        cooked = zeros((1,ns,self.fill_size,nx), Complex)
        cookedIm = vol._subimage(cooked)
        diff = 1000.
        while diff > 50:
            prev_image = cookedIm.data[0].copy()
            cookedIm.data[0] = mag*exp(1.j*theta)
            self.FFT2D.run(cookedIm)
            self.restoreMeasured(cookedIm.data[0], vol.data)
            self.IFFT2D.run(cookedIm)
            diff = sum_flat(abs(cookedIm.data[0] - prev_image))
            print diff
            mag = abs(cookedIm.data[0])
        self.mergeFill(cookedIm.data[0], vol.data, winsize=0)
        return cookedIm.data
    
    def run(self, image):
        (nv, ns, ny, nx) = (image.tdim, image.zdim, image.ydim, image.xdim)
        self.FFT2D = FwdFFT()
        self.IFFT2D = InvFFT()
        self.over_fill = ny - self.fill_size/2
        self.fill_rows = self.fill_size - ny
##    user-foul check should go here        
##         if self.over_fill == 0 or self.fill_dim <= ny:
##             self.log("fill_dim ...
        if self.over_fill < 1:
            self.log("not enough measured data: this method needs a few over-scan lines " \
                     "(lines sampled past the middle of k-space)")
            return
        if self.fill_size <= ny:
            self.log("fill size is not longer than size of measured data "\
                     "(no filling to be done)")
            return

        old_data = image.data.copy()
        old_image = image._subimage(old_data)
        image.data.resize((nv,ns,self.fill_size,nx))
        image.setData(image.data)
        for t in range(nv):
            vol = old_image.subImage(t)

            #phase-map corrected cook:
            #cooked = self.cookImage(vol)

            #iterative cook:
            cooked = self.cookImage2(vol)

            # zero-padded:
            #cooked = self.imageFromFill(vol)
            
##             for s in range(ns):
##                 self.HermitianFill(cooked[0,s], self.fill_rows+8)
                
            image.data[t][:] = cooked[0][:].astype(Complex32)

