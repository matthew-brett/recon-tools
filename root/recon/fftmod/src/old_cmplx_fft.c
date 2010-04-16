#include <fftw3.h>
#include <stdlib.h>

#define INVERSE +1
#define FORWARD -1
#define MAXTHREADS 16
/* Using LAPACK notation for double/single precision complex */
void zfft1d(fftw_complex *zin, fftw_complex *zout, 
	    int len_xform, int len_z, int direction, int shift, int nthreads);
void cfft1d(fftwf_complex *zin, fftwf_complex *zout, 
	    int len_xform, int len_z, int direction, int shift, int nthreads);
void zfft2d(fftw_complex *zin, fftw_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift, int nthreads);
void cfft2d(fftwf_complex *zin, fftwf_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift, int nthreads);

/**************************************************************************
* (z/c)fft1d                                                              *
*                                                                         *
* Repeatedly takes a 1D FFT of length len_xform by advancing the zin and  *
* zout pointers len_z/len_xform times. "direction" indicates whether the  *
* transform is a forward of reverse FFT. Quadrant shifting a la fftshift  *
* in Matlab is done by modulation in both domains. Always normalize ifft. *
**************************************************************************/

void zfft1d(fftw_complex *zin, fftw_complex *zout, 
	    int len_xform, int len_z, int direction, int shift, int nthreads)
{
  fftw_plan FT1D;
  double *dptr1, *dptr2;
  double tog = 1.0, oscl;
  int k, nxforms = len_z/len_xform;

  fftw_plan_with_nthreads(nthreads);
  FT1D = fftw_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform, 
			    zout, NULL, 1, len_xform, direction,
			    FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  // if shifting, modulate the input
  if(shift) {
    dptr1 = (double *) zin;
    for(k=0; k<len_z; k++) {
      // multiply {real,imag} by +/- 1.0;
      *(dptr1++) *= tog;
      *(dptr1++) *= tog;
      tog *= -1.0;
    }
  }
  fftw_execute(FT1D);
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  dptr2 = (double *) zout;
  oscl = (direction==INVERSE) ? 1.0/(double) len_xform : 1.0;
  if(shift) {
    if(zin != zout) {
      dptr1 = (double *) zin;
      for(k=0; k<len_z; k++) {
	*(dptr1++) *= tog;
	*(dptr1++) *= tog;
	tog *= -1.0;
      }
    }
    for(k=0; k<len_z; k++) {
      *(dptr2++) *= oscl;
      *(dptr2++) *= oscl;
      oscl *= -1.0;
    }
  } else if(direction == INVERSE) {
    for(k=0; k<len_z; k++) {
      *(dptr2++) *= oscl;
      *(dptr2++) *= oscl;
    }
  }
  fftw_destroy_plan(FT1D);
  //fftw_cleanup();
}

typedef struct {
  fftwf_complex *i;
  fftwf_complex *o;
  fftwf_plan p;
  int len;
} fftwf_args;

void *fftwf_call(fftwf_args *args)
{
  int k;
  float *fptr1, *fptr2;
  float tog, oscl;

  

  fftwf_execute_dft(args->p, args->i, args->o);

}

void cfft1d(fftwf_complex *zin, fftwf_complex *zout, 
	    int len_xform, int len_z, int direction, int shift, int nthreads)
{
  fftwf_plan FT1D;
  float *fptr1, *fptr2;
  float tog = 1.0, oscl;
  int k, nxforms = len_z/len_xform/nthreads;

  pthread_t threads[MAXTHREADS];
  fftwf_args *args;
  fftwf_args *arg;
  args = (fftwf_args *) malloc(MAXTHREADS * sizeof(fftwf_args));
  

  FT1D = fftwf_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform, 
			     zout, NULL, 1, len_xform, direction,
			     FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);


  for(k=0; k<nthreads; k++) {
    (args+k)->i = zin + nxforms*k;
    (args+k)->o = zout + nxforms*k;
    (args+k)->p = FT1D;
    (args+k)->len = len_xform*nxforms;
    pthread_create(&(threads[k]), NULL, fftwf_call, (void *) (args+k));
  }
  

  // modulate the input
  if(shift) {
    fptr1 = (float *) zin;
    for(k=0; k<len_z; k++) {
      *(fptr1++) *= tog;
      *(fptr1++) *= tog;
      tog *= -1.0;
    }
  }
  fftwf_execute(FT1D);
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  fptr2 = (float *) zout;
  oscl = (direction==INVERSE) ? 1.0/(float) len_xform : 1.0;
  if(shift) {
    if(zin != zout) {
      fptr1 = (float *) zin;
      for(k=0; k<len_z; k++) {
	*(fptr1++) *= tog;
	*(fptr1++) *= tog;
	tog *= -1.0;
      }
    }
    for(k=0; k<len_z; k++) {
      *(fptr2++) *= oscl;
      *(fptr2++) *= oscl;
      oscl *= -1.0;
    }
  } else if(direction == INVERSE) {
    for(k=0; k<len_z; k++) {
      *(fptr2++) *= oscl;
      *(fptr2++) *= oscl;
    }
  }
  fftwf_destroy_plan(FT1D);
  //fftwf_cleanup();
}


/**************************************************************************
* (z/c)fft2d                                                              *
*                                                                         *
* Repeatedly take a 2D FFT of length xdim*ydim across the input array.    *
* "direction" incidates the sign on the complex exponential's argument.   *
* Quadrant "shifting" a la fftshift in Matlab is done by modulation in    *
* both domains. Always normalize the output of the inverse FFT.           *
**************************************************************************/
void zfft2d(fftw_complex *zin, fftw_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift, int nthreads)
{
  double tog = 1.0, oscl;
  int k, size_xform = xdim*ydim, nxforms = len_z/(xdim*ydim);
  int dims[2] = {ydim, xdim};
  fftw_plan FT2D;
  double *dptr1, *dptr2;
  
  fftw_plan_with_nthreads(nthreads);
  FT2D = fftw_plan_many_dft(2, dims, nxforms, 
			    zin, NULL, 1, size_xform,
			    zout, NULL, 1, size_xform,
			    direction, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  if(shift) {
    dptr1 = (double *) zin;
    for(k=0; k<len_z; k++) {
      *(dptr1++) *= tog;
      *(dptr1++) *= tog;
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  }
  fftw_execute(FT2D);
  tog = 1.0;
  dptr2 = (double *) zout;
  oscl = (direction==INVERSE) ? 1.0/(double) size_xform : 1.0;
  if(shift) {
    if(zin != zout) {
      dptr1 = (double *) zin;
      for(k=0; k<len_z; k++) {
	*(dptr1++) *= tog;
	*(dptr1++) *= tog;
	if( (k+1)%xdim ) {
	  tog *= -1.0;
	}
      }
    }
    for(k=0; k<len_z; k++) {
      *(dptr2++) *= oscl;
      *(dptr2++) *= oscl;
      if( (k+1)%xdim ) {
	oscl *= -1.0;
      }
    }
  } else if(direction==INVERSE) {
    for(k=0; k<len_z; k++) {
      *(dptr2++) *= oscl;
      *(dptr2++) *= oscl;
    }
  }
    
  fftw_destroy_plan(FT2D);
  //fftw_cleanup();
  return;
}	     

void cfft2d(fftwf_complex *zin, fftwf_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift, int nthreads)
{
  float tog = 1.0, oscl;
  int k, size_xform = xdim*ydim, nxforms = len_z/(xdim*ydim);
  int dims[2] = {ydim, xdim};
  fftwf_plan FT2D;
  float *fptr1, *fptr2;

  fftwf_plan_with_nthreads(nthreads);
  FT2D = fftwf_plan_many_dft(2, dims, nxforms, 
			     zin, NULL, 1, size_xform,
			     zout, NULL, 1, size_xform,
			     direction, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  if(shift) {
    fptr1 = (float *) zin;
    for(k=0; k<len_z; k++) {
      *(fptr1++) *= tog;
      *(fptr1++) *= tog;
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  }
  fftwf_execute(FT2D);
  tog = 1.0;
  fptr2 = (float *) zout;
  oscl = (direction==INVERSE) ? 1.0/(float) size_xform : 1.0;
  if(shift) {
    if(zin != zout) {
      fptr1 = (float *) zin;
      for(k=0; k<len_z; k++) {
	*(fptr1++) *= tog;
	*(fptr1++) *= tog;
	if( (k+1)%xdim ) {
	  tog *= -1.0;
	}
      }
    }
    for(k=0; k<len_z; k++) {
      *(fptr2++) *= oscl;
      *(fptr2++) *= oscl;
      if( (k+1)%xdim ) {
	oscl *= -1.0;
      }
    }
  } else if(direction==INVERSE) {
    for(k=0; k<len_z; k++) {
      *(fptr2++) *= oscl;
      *(fptr2++) *= oscl;
    }
  }

  fftwf_destroy_plan(FT2D);
  //fftwf_cleanup();
  return;
}	     
