
#include "Python.h"
#include "numpy/noprefix.h"
#include <fftw3.h>
#include <stdlib.h>

#define INVERSE +1
#define FORWARD -1
/* Using LAPACK notation for double/single precision complex */
void zfft1d(fftw_complex *zin, fftw_complex *zout, 
	    int len_xform, int len_z, int direction, int shift);
void cfft1d(fftwf_complex *zin, fftwf_complex *zout, 
	    int len_xform, int len_z, int direction, int shift);
void zfft2d(fftw_complex *zin, fftw_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift);
void cfft2d(fftwf_complex *zin, fftwf_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift);

static char doc_fft1d[] = "Performs the (I)FFT repeatedly on the last dimension of a multidimensional array";

PyObject *fftmod_fft1d(PyObject *self, PyObject *args) {
  PyObject *op1;
  PyArrayObject *inArray, *retArray;
  fftw_complex *z1=NULL, *z2=NULL;
  fftwf_complex *c1=NULL, *c2=NULL;
  int type, dir, shift, len_xform, len_array, ndim;
  npy_intp *dims;
  PyTypeObject *pytype;
  PyArray_Descr *dtype;

  if(!PyArg_ParseTuple(args, "Oii", &op1, &dir, &shift)) {
    printf("error parsing args in fft1d()\n");
    return NULL;
  }
  type = PyArray_TYPE(op1);
  if(type != PyArray_CFLOAT && type != PyArray_CDOUBLE) {
    PyErr_SetString(PyExc_TypeError, "Attempted a complex FFT on non-complex data\n");
    return NULL;
  }
  ndim = PyArray_NDIM(op1);
  if (ndim < 1) {
    PyErr_SetString(PyExc_Exception, "Can't transform a 0-d array");
    return NULL;
  }

  dtype = PyArray_DescrFromType(type);
  /* This gives a PyArrayObject with the correct type and contiguous data */
  /* It also increases the reference count! */
  inArray = (PyArrayObject *)PyArray_FROM_OTF(op1, type, NPY_IN_ARRAY);
  len_array = PyArray_SIZE(inArray);
  dims = PyArray_DIMS(inArray);
  /* work on the last axis of the data */
  len_xform = (int) dims[ndim-1];

  retArray = (PyArrayObject *)PyArray_SimpleNewFromDescr(ndim, dims, dtype);
  if(type == PyArray_CFLOAT) {
    c1 = (fftwf_complex *)PyArray_DATA(inArray);
    c2 = (fftwf_complex *)PyArray_DATA(retArray);
    cfft1d(c1, c2, len_xform, len_array, dir, shift);
  } else {
    z1 = (fftw_complex *)PyArray_DATA(inArray);
    z2 = (fftw_complex *)PyArray_DATA(retArray);
    zfft1d(z1, z2, len_xform, len_array, dir, shift);
  }
  Py_DECREF(inArray);
  return PyArray_Return(retArray);

}

static char doc_fft2d[] = "Performs the (I)FFT-2D repeatedly on the last 2 dimensions of a multidimensional array";

PyObject *fftmod_fft2d(PyObject *self, PyObject *args) {
  PyObject *op1;
  PyArrayObject *inArray, *retArray;
  fftw_complex *z1=NULL, *z2=NULL;
  fftwf_complex *c1=NULL, *c2=NULL;
  int type, dir, shift, xdim, ydim, ndim, len_array;
  npy_intp *dims;
  PyTypeObject *pytype;
  PyArray_Descr *dtype;
  
  if(!PyArg_ParseTuple(args, "Oii", &op1, &dir, &shift)) {
    printf("error parsing args in fft2d()\n");
    return NULL;
  }
  type = PyArray_ObjectType(op1, 0);
  if(type != PyArray_CFLOAT && type != PyArray_CDOUBLE) {
    PyErr_SetString(PyExc_TypeError, "Attempted a complex FFT2D on non-complex data\n");
    return NULL;
  }
  
  ndim = PyArray_NDIM(op1);
  if (ndim < 2) {
    PyErr_SetString(PyExc_Exception, "Can't transform a 1-d array");
    return NULL;
  }

  dtype = PyArray_DescrFromType(type);
  /* This gives a PyArrayObject with the correct type and contiguous data */
  /* It also increases the reference count! */
  inArray = (PyArrayObject *)PyArray_FROM_OTF(op1, type, NPY_IN_ARRAY);
  len_array = PyArray_SIZE(inArray);
  dims = PyArray_DIMS(inArray);
  xdim = dims[ndim-1];   // n_fe
  ydim = dims[ndim-2];   // n_pe
  
  retArray = (PyArrayObject *)PyArray_SimpleNewFromDescr(ndim, dims, dtype);
  
  if(type == PyArray_CFLOAT) {
    c1 = (fftwf_complex *)PyArray_DATA(inArray);
    c2 = (fftwf_complex *)PyArray_DATA(retArray);
    cfft2d(c1, c2, xdim, ydim, len_array, dir, shift);
  } else {
    z1 = (fftw_complex *)PyArray_DATA(inArray);
    z2 = (fftw_complex *)PyArray_DATA(retArray);
    zfft2d(z1, z2, xdim, ydim, len_array, dir, shift);
  }
  Py_DECREF(inArray);
  return PyArray_Return(retArray);
}
   
/**************************************************************************
* (z/c)fft1d                                                              *
*                                                                         *
* Repeatedly takes a 1D FFT of length len_xform by advancing the zin and  *
* zout pointers len_z/len_xform times. "direction" indicates whether the  *
* transform is a forward of reverse FFT. Quadrant shifting a la fftshift  *
* in Matlab is done by modulation in both domains. Always normalize ifft. *
**************************************************************************/

void zfft1d(fftw_complex *zin, fftw_complex *zout, 
	    int len_xform, int len_z, int direction, int shift)
{
  fftw_plan FT1D;
  fftw_complex *zptr1=NULL, *zptr2=NULL;
  double tog = 1.0, alpha;
  int k, nxforms = len_z/len_xform;

  FT1D = fftw_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform,
			    zout, NULL, 1, len_xform, direction,
			    FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  // if shifting, modulate the input
  if(shift) {
    zptr1 = zin;
    for(k=0; k<len_z; k++) {
      *((double *) (zptr1)) *= tog;
      *((double *) (zptr1++) + 1) *= tog;
      tog *= -1.0;
    }
  }
  fftw_execute(FT1D);
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  zptr2 = zout;
  if(shift) {
    zptr1 = (zin != zout) ? zin : NULL;
    for(k=0; k<len_z; k++) {
      if(zptr1) {
	*((double *) (zptr1)) *= tog;
	*((double *) (zptr1++) + 1) *= tog;
      }
      alpha = (direction==INVERSE) ? tog/(double) len_xform : tog;
      *((double *) (zptr2)) *= alpha;
      *((double *) (zptr2++) + 1) *= alpha;
      tog *= -1.0;
    }
  } else {
    if(direction == INVERSE) {
      alpha = 1.0 / (double) len_xform;
      for(k=0; k<len_z; k++) {
	*((double *) (zptr2)) *= alpha;
	*((double *) (zptr2++) + 1) *= alpha;
      }
    }
  }
  fftw_destroy_plan(FT1D);
  fftw_cleanup();
}

void cfft1d(fftwf_complex *zin, fftwf_complex *zout, 
	    int len_xform, int len_z, int direction, int shift)
{
  fftwf_plan FT1D;
  fftwf_complex *cptr1=NULL, *cptr2=NULL;
  float tog = 1.0, alpha;
  int k, nxforms = len_z/len_xform;
  FT1D = fftwf_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform,
			     zout, NULL, 1, len_xform, direction,
			     FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  // modulate the input
  if(shift) {
    cptr1 = zin;
    for(k=0; k<len_z; k++) {
      *((float *) (cptr1)) *= tog;
      *((float *) (cptr1++) + 1) *= tog;
      tog *= -1.0;
    }
  }
  fftwf_execute(FT1D);
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  cptr2 = zout;
  if(shift) {
    cptr1 = (zin != zout) ? zin : NULL;
    for(k=0; k<len_z; k++) {
      if(cptr1) {
	*((float *) (cptr1)) *= tog;
	*((float *) (cptr1++) + 1) *= tog;	
      }
      alpha = (direction==INVERSE) ? tog/(float) len_xform : tog;
      *((float *) (cptr2)) *= alpha;
      *((float *) (cptr2++) + 1) *= alpha;
      tog *= -1.0;
    }
  } else {
    if(direction == INVERSE) {
      alpha = 1.0 / (float) len_xform;
      for(k=0; k<len_z; k++) {
	*((float *) (cptr2)) *= alpha;
	*((float *) (cptr2++) + 1) *= alpha;
      }
    }
  }
  fftwf_destroy_plan(FT1D);
  fftwf_cleanup();
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
	    int len_z, int direction, int shift)
{
  double tog = 1.0, alpha;
  int k, size_xform = xdim*ydim, nxforms = len_z/(xdim*ydim);
  int dims[2] = {ydim, xdim};
  fftw_plan FT2D;
  fftw_complex *zptr1, *zptr2;
  
  FT2D = fftw_plan_many_dft(2, dims, nxforms, 
			    zin, NULL, 1, size_xform,
			    zout, NULL, 1, size_xform,
			    direction, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  if(shift) {
    zptr1 = zin;
    for(k=0; k<len_z; k++) {
      *((double *) (zptr1)) *= tog;
      *((double *) (zptr1++) + 1) *= tog;
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  }
  fftw_execute(FT2D);
  tog = 1.0;
  zptr2 = zout;
  if(shift) {
    zptr1 = (zin != zout) ? zin : NULL;
    for(k=0; k<len_z; k++) {
      if(zptr1) {
	*((double *) (zptr1)) *= tog;
	*((double *) (zptr1++) + 1) *= tog;
      }
      alpha = (direction==INVERSE) ? tog/(double) size_xform : tog;
      *((double *) (zptr2)) *= alpha;
      *((double *) (zptr2++) + 1) *= alpha;
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  } else {
    if(direction==INVERSE) {
      alpha = 1.0 / (double) size_xform;
      for(k=0; k<len_z; k++) {
	*((double *) (zptr2)) *= alpha;
	*((double *) (zptr2++) + 1) *= alpha;
      }
    }
  }
    
  fftw_destroy_plan(FT2D);
  fftw_cleanup();
  return;
}	     

void cfft2d(fftwf_complex *zin, fftwf_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift)
{
  float tog = 1.0, alpha;
  int k, size_xform = xdim*ydim, nxforms = len_z/(xdim*ydim);
  int dims[2] = {ydim, xdim};
  fftwf_plan FT2D;
  fftwf_complex *cptr1, *cptr2;

  FT2D = fftwf_plan_many_dft(2, dims, nxforms, 
			     zin, NULL, 1, size_xform,
			     zout, NULL, 1, size_xform,
			     direction, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  if(shift) {
    cptr1 = zin;
    for(k=0; k<len_z; k++) {
      *((float *) (cptr1)) *= tog;
      *((float *) (cptr1++) + 1) *= tog;
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  }
  fftwf_execute(FT2D);
  tog = 1.0;
  cptr2 = zout;
  if(shift) {
    cptr1 = (zin != zout) ? zin : NULL;
    for(k=0; k<len_z; k++) {
      if(cptr1) {
	*((float *) (cptr1)) *= tog;
	*((float *) (cptr1++) + 1) *= tog;
      }
      alpha = (direction==INVERSE) ? tog/(float) size_xform : tog;
      *((float *) (cptr2)) *= alpha;
      *((float *) (cptr2++) + 1) *= alpha;
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  } else {
    if(direction==INVERSE) {
      alpha = 1.0 / (float) size_xform;
      for(k=0; k<len_z; k++) {
	*((float *) (cptr2)) *= alpha;
	*((float *) (cptr2++) + 1) *= alpha;
      }
    }
  }

  fftwf_destroy_plan(FT2D);
  fftwf_cleanup();
  return;
}	     


static struct PyMethodDef fftmod_module_methods[] = {
  {"fft1d", (PyCFunction)fftmod_fft1d, 1, doc_fft1d},
  {"fft2d", (PyCFunction)fftmod_fft2d, 1, doc_fft2d},
  {NULL, NULL, 0}
};

DL_EXPORT(void) init_fftmod(void) {
  PyObject *m;
  m = Py_InitModule3("_fftmod", fftmod_module_methods, "C library for complex FFT");
  import_array();
}
