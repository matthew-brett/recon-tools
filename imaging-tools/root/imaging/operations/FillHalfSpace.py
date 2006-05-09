import sys
from imaging.operations import Operation, Parameter
from pylab import zeros, Complex32, Float, arange, reshape, rot90, conjugate, angle, sum_flat, exp, Complex, NewAxis, cos, pi
from imaging.util import embedIm, checkerboard
from FFT import inverse_fft2d, fft2d
from imaging.operations.ForwardFFT import ForwardFFT as FwdFFT
from imaging.operations.InverseFFT import InverseFFT as InvFFT



class FillHalfSpace (Operation):

    params=(
        Parameter(name="fill_size", type="int", default=0,
                  description="the new number of rows to fill to in k-space"),
        Parameter(name="win_size", type="int", default=8,
                  description="length of transition window between measured "\
                  "k-space and filled k-space; a window reduces Gibbs ringing"),
        Parameter(name="iterations", type="int", default=0,
                  description="number of times to iterate the merge process"),
        Parameter(name="converge_crit", type="float", default=0,
                  description="stop iteration when the summed absolute " \
                  "difference between sucessive reconstructed volumes equals "\
                  "this amount"),
        Parameter(name="method", type="str", default="by volume",
                  description="possible values: iterative, zero filled")
        )

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

    def phaseMap2D(self, slice):
        (_, nx) = slice.shape
        ny = self.fill_size
        y0 = self.fill_size/2
        fill_slice = zeros((ny,nx), Complex)
        fill_slice[y0-self.over_fill:y0+self.over_fill,:] = \
                   slice[0:self.over_fill*2,:]
        #phase_map = angle(inverse_fft2d(fill_slice))
        phase_map = angle(self.mask*inverse_fft2d(self.mask*fill_slice))
        return phase_map
    

    def imageFromFill(self, vol):
        (ns, _, nx) = vol.data.shape
        ny = self.fill_size
        fill_vol = zeros((1,ns,ny,nx), Complex32)
        for s in range(ns):
            embedIm(vol.data[s], fill_vol[0,s], self.fill_rows, 0)
        fill_image = vol._subimage(fill_vol)
        self.IFFT2D.run(fill_image)
        return fill_image.data[0]

    def imageFromFill2D(self, slice):
        (_, nx) = slice.shape
        ny = self.fill_size
        fill_slice = zeros((ny,nx), Complex)
        embedIm(slice, fill_slice, self.fill_rows, 0)
        #fill_slice[:] = inverse_fft2d(fill_slice)
        fill_slice[:] = self.mask*inverse_fft2d(self.mask*fill_slice)
        return fill_slice

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

    def restoreMeasured(self, filled, measured):
        filled[:,self.fill_rows:,:] = measured[:].astype(filled.typecode())

    def mergeFill(self, filled, measured, winsize=8):
        mergept = self.fill_rows
        fill_win = 0.5*(1 + cos(pi*(arange(winsize)/float(winsize))))
        measured_win = 0.5*(1 + cos(pi + pi*(arange(winsize)/float(winsize))))
        # put back original measured data in rows mergept+winsize and on
        #filled[:,mergept+winsize:,:] = measured[:,winsize:,:]
        # merge measured data with filled data in winsize merge region
        filled[:,mergept:mergept+winsize,:] = \
               fill_win[:,NewAxis]*filled[:,mergept:mergept+winsize,:] + \
               measured_win[:,NewAxis]*measured[:,:winsize,:]

    def mergeFill2D(self, filled, measured, winsize=8):
        from pylab import sqrt
        mergept = self.fill_rows
        fill_win = 0.5*(1 + cos(pi*(arange(winsize)/float(winsize))))
        measured_win = 0.5*(1 + cos(pi + pi*(arange(winsize)/float(winsize))))
        filled[:mergept,:] = 1.5*filled[:mergept,:]
        filled[mergept+winsize:,:] = measured[winsize:,:]
        # merge measured data with filled data in winsize merge region
        filled[mergept:mergept+winsize,:] = \
               fill_win[:,NewAxis]*filled[mergept:mergept+winsize,:] + \
               measured_win[:,NewAxis]*measured[:winsize,:]

    def cookImage(self, vol):
        theta = self.phaseMap(vol)
        mag = abs(self.imageFromFill(vol))
        (ns, _, nx) = vol.data.shape
        cooked = zeros((1,ns,self.fill_size,nx), Complex)
        cookedIm = vol._subimage(cooked)
        diff = 50000.
        while diff > self.converge_crit:
        #while self.iterations > 0:
            prev_image = cookedIm.data[0].copy()
            cookedIm.data[0] = mag*exp(1.j*theta)
            self.FFT2D.run(cookedIm)
            cookedIm.data[0] = cookedIm.data[0]
            self.restoreMeasured(cookedIm.data[0], vol.data)            
            self.IFFT2D.run(cookedIm)
            diff = sum_flat(abs(cookedIm.data[0] - prev_image))
            print diff
            mag = abs(cookedIm.data[0])
            #self.iterations -= 1
        self.FFT2D.run(cookedIm)
        self.mergeFill(cookedIm.data[0], vol.data, winsize=self.win_size)
        return cookedIm.data[0].astype(vol.data.typecode())

    def cookImage2D(self, volData):
        from pylab import imshow, show, colorbar
        out = sys.stdout
        (ns, _, nx) = volData.shape
        ny = self.fill_size
        cooked3D = zeros((ns,ny,nx), Complex)
        for s, slice in enumerate(volData):
            out.write("filling slice %d: "%(s,))
            theta = self.phaseMap2D(slice)
            mag = 1.5*abs(self.imageFromFill2D(slice))
            cooked = zeros((ny, nx), Complex)
            c = self.criterion[1]=="converge" and 10000. or self.iterations
            while c > self.criterion[0]:
                prev_image = cooked.copy()
                ## imshow(self.mask*fft2d(self.mask*mag))
##                 #imshow(mag)
##                 colorbar()
##                 show()
##                 imshow(self.mask*fft2d(self.mask*exp(1.j*theta)))
##                 #imshow(theta)
##                 colorbar()
##                 show()
                cooked = mag*exp(1.j*theta)
                cooked[:] = self.mask*fft2d(self.mask*cooked)
##                 imshow(cooked)
##                 colorbar()
##                 show()
                cooked[self.fill_rows:,:] = slice[:]
                cooked[:] = self.mask*inverse_fft2d(self.mask*cooked)
##                 imshow(abs(cooked))
##                 colorbar()
##                 show()
                diff = sum_flat(abs(cooked-prev_image))
                mag = abs(cooked)
                c = self.criterion[1]=="converge" and diff or c-1
            cooked[:] = self.mask*fft2d(self.mask*cooked)
            self.mergeFill2D(cooked, slice, winsize=self.win_size)
            cooked3D[s][:] = cooked[:]
            out.write("absolute difference=%f\n"%(diff))
        return cooked3D.astype(volData.typecode())
    
    def run(self, image):
        (nv, ns, ny, nx) = (image.tdim, image.zdim, image.ydim, image.xdim)
        self.FFT2D = FwdFFT()
        self.IFFT2D = InvFFT()
        self.over_fill = ny - self.fill_size/2
        self.fill_rows = self.fill_size - ny
        self.mask = checkerboard(self.fill_size, nx)
        if self.over_fill < 1:
            self.log("not enough measured data: this method needs a few " \
                     "over-scan lines (sampled past the middle of k-space)")
            return
        if self.fill_size <= ny:
            self.log("fill size is not longer than size of measured data "\
                     "(no filling to be done)")
            return

        if self.converge_crit > 0 and self.iterations > 0:
            self.log("you cannot specify the convergence criterion OR the "\
                     "number of iterations, NOT both: doing nothing")
            return
        elif self.converge_crit > 0:
            self.criterion = (self.converge_crit,"converge")
        elif self.iterations > 0:
            self.criterion = (0,"iterateN")
        else:
            self.log("no iterative criterion given, default to 5 iterations")
            self.criterion = (0,"iterateN")
            self.iterations = 5
        
        old_data = image.data.copy()
        old_image = image._subimage(old_data)
        image.data.resize((nv,ns,self.fill_size,nx))
        image.setData(image.data)

        for t in range(nv):
            vol = old_image.subImage(t)
            
            if self.method == "iterative":
                cooked = self.cookImage2D(vol.data)

##             elif self.method == "by volume":
##                 cooked = self.cookImage(vol)
                
            else:
                cooked = self.kSpaceFill(vol)
                
            image.data[t][:] = cooked[:]
            
    def kSpaceFill(self, vol):
        (ns, _, nx) = vol.data.shape
        ny = self.fill_size
        fill_vol = zeros((ns,ny,nx), Complex32)
        for s in range(ns):
            embedIm(vol.data[s], fill_vol[s], self.fill_rows, 0)
        return fill_vol.astype(vol.data.typecode())
