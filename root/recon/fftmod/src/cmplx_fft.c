#include <fftw3.h>
#include <stdlib.h>
#include <pthread.h>

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

void *thread1D_fftw(void *ptr);
void *thread1D_fftwf(void *ptr);
void *thread2D_fftw(void *ptr);
void *thread2D_fftwf(void *ptr);

// probably can make just one type
typedef struct {
  fftwf_complex *i;
  fftwf_complex *o;
  fftwf_plan p;
  int n_xform;
  int len_xform;
  int shift;
  int direction;
  int dims[3];
} fftwf_args;

typedef struct {
  fftw_complex *i;
  fftw_complex *o;
  fftw_plan p;
  int n_xform;
  int len_xform;
  int shift;
  int direction;
  int dims[3];
} fftw_args;

typedef struct {
  char *i;
  char *o;
  char *p;
  int n_xform;
  int len_xform;
  int shift;
  int direction;
  int dims[3];
} args_list;

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
  int k, nxforms;
  args_list *args;
/*   args = (fftw_args *) malloc(MAXTHREADS * sizeof(fftw_args)); */
  args = (args_list *) malloc(MAXTHREADS * sizeof(args_list));

  if( (len_z/len_xform) % nthreads ) {
    nthreads = 1;
  }   
  nxforms = len_z/len_xform/nthreads;
  FT1D = fftw_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform, 
			    zout, NULL, 1, len_xform, direction,
			    FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

#ifdef THREADED // THIS IS THE THREADED BLOCK
  pthread_t threads[MAXTHREADS];

  // total block size = len_xform*nxforms*nthreads = len_z
  // stride through the memory as k*(len_xform*nxforms), k = 0,1,...,nthreads-1
  for(k=0; k<nthreads; k++) {
    (args+k)->i = (char *) (zin + len_xform*nxforms*k);
    (args+k)->o = (char *) (zout + len_xform*nxforms*k);
    (args+k)->p = (char *) &FT1D;
    (args+k)->n_xform = nxforms;
    (args+k)->len_xform = len_xform;
    (args+k)->shift = shift;
    (args+k)->direction = direction;
    pthread_create(&(threads[k]), NULL, thread1D_fftw, (void *) (args+k));
  }


  for(k=0; k<nthreads; k++) {
    pthread_join(threads[k], NULL);
  }

#else // THIS IS THE MONO-THREAD BLOCK

  args->i = (char *) zin;
  args->o = (char *) zout;
  args->p = (char *) &FT1D;
  args->n_xform = nxforms;
  args->len_xform = len_xform;
  args->shift = shift;
  args->direction = direction;
  thread1D_fftw((void *) args);

#endif

  fftw_destroy_plan(FT1D);
  //fftw_cleanup();
}

void *thread1D_fftw(void *ptr)
{
  int k, len_z;
  double *dptr1, *dptr2;
  double tog, oscl;
  fftw_plan *FT1D;
  //fftw_args *args = (fftw_args *) ptr;
  args_list *args = (args_list *) ptr;

  len_z = args->n_xform * args->len_xform;

  // if shifting, modulate the input
  tog = 1.0;
  if(args->shift) {
    dptr1 = (double *) args->i;
    for(k=0; k<len_z; k++) {
      // multiply {real,imag} by +/- 1.0;
      *(dptr1++) *= tog;
      *(dptr1++) *= tog;
      tog *= -1.0;
    }
  }
  FT1D = (fftw_plan *) args->p;
#ifdef THREADED
  fftw_execute_dft(*FT1D, (fftw_complex *) args->i, (fftw_complex *) args->o);
#else
  fftw_execute(*FT1D);
#endif
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  dptr2 = (double *) args->o;
  oscl = (args->direction==INVERSE) ? 1.0/(double) args->len_xform : 1.0;
  if(args->shift) {
    if(args->i != args->o) {
      dptr1 = (double *) args->i;
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
  } else if(args->direction == INVERSE) {
    for(k=0; k<len_z; k++) {
      *(dptr2++) *= oscl;
      *(dptr2++) *= oscl;
    }
  }
}

void cfft1d(fftwf_complex *zin, fftwf_complex *zout, 
	    int len_xform, int len_z, int direction, int shift, int nthreads)
{
  fftwf_plan FT1D;
  int k, nxforms;
  args_list *args;
  //args = (fftwf_args *) malloc(MAXTHREADS * sizeof(fftwf_args));
  args = (args_list *) malloc(MAXTHREADS * sizeof(args_list));
  
  if( (len_z/len_xform) % nthreads ) {
    nthreads = 1;
  }   
  
  nxforms = len_z/len_xform/nthreads;
  FT1D = fftwf_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform, 
			     zout, NULL, 1, len_xform, direction,
			     FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

#ifdef THREADED
  pthread_t threads[MAXTHREADS];
  // total block size = len_xform*nxforms*nthreads = len_z
  // stride through the memory as k*(len_xform*nxforms), k = 0,1,...,nthreads-1
  for(k=0; k<nthreads; k++) {
    (args+k)->i = (char *) (zin + len_xform*nxforms*k);
    (args+k)->o = (char *) (zout + len_xform*nxforms*k);
    (args+k)->p = (char *) &FT1D;
    (args+k)->n_xform = nxforms;
    (args+k)->len_xform = len_xform;
    (args+k)->shift = shift;
    (args+k)->direction = direction;
    pthread_create(&(threads[k]), NULL, thread1D_fftwf, (void *) (args+k));
  }


  for(k=0; k<nthreads; k++) {
    pthread_join(threads[k], NULL);
  }

#else
  args->i = (char *) zin;
  args->o = (char *) zout;
  args->p = (char *) &FT1D;
  args->n_xform = nxforms;
  args->len_xform = len_xform;
  args->shift = shift;
  args->direction = direction;
  thread1D_fftwf((void *) args);

#endif

  fftwf_destroy_plan(FT1D);
  //fftwf_cleanup();
}

void *thread1D_fftwf(void *ptr)
{
  int k, len_z;
  float *fptr1, *fptr2;
  float tog, oscl;
  fftwf_plan *FT1D;
  args_list *args = (args_list *) ptr;

  len_z = args->n_xform * args->len_xform;
  
  // modulate the input
  tog = 1.0;
  if(args->shift) {
    fptr1 = (float *) args->i;
    for(k=0; k<len_z; k++) {
      *(fptr1++) *= tog;
      *(fptr1++) *= tog;
      tog *= -1.0;
    }
  }
  FT1D = (fftwf_plan *) args->p;
#ifdef THREADED
  fftwf_execute_dft(*FT1D, (fftwf_complex *)args->i, (fftwf_complex *)args->o);
#else
  fftwf_execute(*FT1D);
#endif
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  fptr2 = (float *) args->o;
  oscl = (args->direction==INVERSE) ? 1.0/(float) args->len_xform : 1.0;
  if(args->shift) {
    if(args->i != args->o) {
      fptr1 = (float *) args->i;
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
  } else if(args->direction == INVERSE) {
    for(k=0; k<len_z; k++) {
      *(fptr2++) *= oscl;
      *(fptr2++) *= oscl;
    }
  }
  
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
  int k, len_xform = xdim*ydim, nxforms;
  int dims[2] = {ydim, xdim};
  fftw_plan FT2D;
  args_list *args;
  args = (args_list *) malloc(MAXTHREADS * sizeof(args_list));

  if( (len_z/(xdim*ydim)) % nthreads ) {
    nthreads = 1;
  }
  nxforms = len_z/(xdim*ydim)/nthreads;

  FT2D = fftw_plan_many_dft(2, dims, nxforms, 
			    zin, NULL, 1, len_xform,
			    zout, NULL, 1, len_xform,
			    direction, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

#ifdef THREADED
  pthread_t threads[MAXTHREADS];

  for(k=0; k<nthreads; k++) {
    (args+k)->i = (char *) (zin + len_xform*nxforms*k);
    (args+k)->o = (char *) (zout + len_xform*nxforms*k);
    (args+k)->p = (char *) &FT2D;
    (args+k)->n_xform = nxforms;
    (args+k)->len_xform = len_xform;
    (args+k)->shift = shift;
    (args+k)->direction = direction;
    (args+k)->dims[0] = xdim;
    (args+k)->dims[1] = ydim;
    pthread_create(&(threads[k]), NULL, thread2D_fftw, (void *) (args+k));
  }
  for(k=0; k<nthreads; k++) {
    pthread_join(threads[k], NULL);
  }

#else // EXECUTE JUST ONE THREAD
  args->i = (char *) zin;
  args->o = (char *) zout;
  args->p = (char *) &FT2D;
  args->n_xform = nxforms;
  args->len_xform = len_xform;
  args->shift = shift;
  args->direction = direction;
  args->dims[0] = xdim;
  args->dims[1] = ydim;
  thread2D_fftw((void *) args);
#endif

  fftw_destroy_plan(FT2D);
  return;
}	     

void *thread2D_fftw(void *ptr)
{
  int k, len_z;
  double tog, oscl;
  double *dptr1, *dptr2;
  fftw_plan *FT2D;
  args_list *args = (args_list *) ptr;

  len_z = args->n_xform * args->len_xform;
  tog = 1.0;
  if(args->shift) {
    dptr1 = (double *) args->i;
    for(k=0; k<len_z; k++) {
      *(dptr1++) *= tog;
      *(dptr1++) *= tog;
      if( (k+1)%args->dims[0] ) {
	tog *= -1.0;
      }
    }
  }
  FT2D = (fftw_plan *) args->p;
#ifdef THREADED
  fftw_execute_dft(*FT2D, (fftw_complex *) args->i, (fftw_complex *) args->o);
#else
  fftw_execute(*FT2D);
#endif
  tog = 1.0;
  dptr2 = (double *) args->o;
  oscl = (args->direction==INVERSE) ? 1.0/(double) args->len_xform : 1.0;
  if(args->shift) {
    if(args->i != args->o) {
      dptr1 = (double *) args->i;
      for(k=0; k<len_z; k++) {
	*(dptr1++) *= tog;
	*(dptr1++) *= tog;
	if( (k+1)%args->dims[0] ) {
	  tog *= -1.0;
	}
      }
    }
    for(k=0; k<len_z; k++) {
      *(dptr2++) *= oscl;
      *(dptr2++) *= oscl;
      if( (k+1)%args->dims[0] ) {
	oscl *= -1.0;
      }
    }
  } else if(args->direction==INVERSE) {
    for(k=0; k<len_z; k++) {
      *(dptr2++) *= oscl;
      *(dptr2++) *= oscl;
    }
  }
}

void cfft2d(fftwf_complex *zin, fftwf_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift, int nthreads)
{
  int k, len_xform = xdim*ydim, nxforms;
  int dims[2] = {ydim, xdim};
  fftwf_plan FT2D;
  args_list *args;
  args = (args_list *) malloc(MAXTHREADS * sizeof(args_list));

  if( (len_z/(xdim*ydim)) % nthreads ) {
    nthreads = 1;
  }
  nxforms = len_z/(xdim*ydim)/nthreads;

  FT2D = fftwf_plan_many_dft(2, dims, nxforms, 
			     zin, NULL, 1, len_xform,
			     zout, NULL, 1, len_xform,
			     direction, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

#ifdef THREADED
  pthread_t threads[MAXTHREADS];

  for(k=0; k<nthreads; k++) {
    (args+k)->i = (char *) (zin + len_xform*nxforms*k);
    (args+k)->o = (char *) (zout + len_xform*nxforms*k);
    (args+k)->p = (char *) &FT2D;
    (args+k)->n_xform = nxforms;
    (args+k)->len_xform = len_xform;
    (args+k)->shift = shift;
    (args+k)->direction = direction;
    (args+k)->dims[0] = xdim;
    (args+k)->dims[1] = ydim;
    pthread_create(&(threads[k]), NULL, thread2D_fftwf, (void *) (args+k));
  }
  for(k=0; k<nthreads; k++) {
    pthread_join(threads[k], NULL);
  }

#else // EXECUTE JUST ONE THREAD
  args->i = (char *) zin;
  args->o = (char *) zout;
  args->p = (char *) &FT2D;
  args->n_xform = nxforms;
  args->len_xform = len_xform;
  args->shift = shift;
  args->direction = direction;
  args->dims[0] = xdim;
  args->dims[1] = ydim;
  thread2D_fftwf((void *) args);
#endif
  
  fftwf_destroy_plan(FT2D);
  return;
}	     

void *thread2D_fftwf(void *ptr)
{
  int k, len_z;
  float tog, oscl;
  float *fptr1, *fptr2;
  fftwf_plan *FT2D;
  args_list *args = (args_list *) ptr;

  len_z = args->n_xform * args->len_xform;
  tog = 1.0;
  if(args->shift) {
    fptr1 = (float *) args->i;
    for(k=0; k<len_z; k++) {
      *(fptr1++) *= tog;
      *(fptr1++) *= tog;
      if( (k+1)%args->dims[0] ) {
	tog *= -1.0;
      }
    }
  }
  FT2D = (fftwf_plan *) args->p;
#ifdef THREADED
  fftwf_execute_dft(*FT2D, (fftwf_complex *)args->i, (fftwf_complex *)args->o);
#else
  fftwf_execute(*FT2D);
#endif
  tog = 1.0;
  fptr2 = (float *) args->o;
  oscl = (args->direction==INVERSE) ? 1.0/(float) args->len_xform : 1.0;
  if(args->shift) {
    if(args->i != args->o) {
      fptr1 = (float *) args->i;
      for(k=0; k<len_z; k++) {
	*(fptr1++) *= tog;
	*(fptr1++) *= tog;
	if( (k+1)%args->dims[0] ) {
	  tog *= -1.0;
	}
      }
    }
    for(k=0; k<len_z; k++) {
      *(fptr2++) *= oscl;
      *(fptr2++) *= oscl;
      if( (k+1)%args->dims[0] ) {
	oscl *= -1.0;
      }
    }
  } else if(args->direction==INVERSE) {
    for(k=0; k<len_z; k++) {
      *(fptr2++) *= oscl;
      *(fptr2++) *= oscl;
    }
  }
}
