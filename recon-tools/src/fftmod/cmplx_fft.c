
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
  PyArrayObject *ap1, *ret;
  int type, dir, shift, len_xform, len_array, dim;
  PyTypeObject *pytype;
  PyArray_Descr *dtype;

  if(!PyArg_ParseTuple(args, "Oii", &op1, &dir, &shift)) {
    printf("error parsing args in fft1d()\n");
    return NULL;
  }
  type = PyArray_ObjectType(op1, 0);
  if(type != PyArray_CFLOAT && type != PyArray_CDOUBLE) {
    PyErr_SetString(PyExc_TypeError, "Attempted a complex FFT on non-complex data\n");
    return NULL;
  }
  dtype = PyArray_DescrFromType(type);
  // this presents a PyArrayObject with the correct type and contiguous data
  ap1 = (PyArrayObject *)PyArray_FROM_OTF(op1, type, NPY_IN_ARRAY);
  if(ap1==NULL) return NULL;
  pytype = ap1->ob_type;
  if (ap1->nd < 1) {
    PyErr_SetString(PyExc_ValueError, "Can't transform a 0-d array");
    Py_XDECREF(ap1);
    return NULL;
  }
  dim = ap1->nd-1;
  len_xform = ap1->dimensions[dim];
  len_array = 1;
  while(dim >= 0) len_array *= ap1->dimensions[dim--];
  //ret = (PyArrayObject *)PyArray_Empty(ap1->nd, ap1->dimensions, dtype, 0);
  ret = (PyArrayObject *)PyArray_New(pytype, ap1->nd, ap1->dimensions,
				     type, NULL, NULL, 0, 0, (PyObject *) ap1);
  if(type == PyArray_CFLOAT)
    cfft1d((fftwf_complex *)ap1->data, (fftwf_complex *)ret->data, 
	   len_xform, len_array, dir, shift);
  else
    zfft1d((fftw_complex *)ap1->data, (fftw_complex *)ret->data,
	   len_xform, len_array, dir, shift);
  
  Py_DECREF(ap1);
  return PyArray_Return(ret);

}

static char doc_fft2d[] = "Performs the (I)FFT-2D repeatedly on the last 2 dimensions of a multidimensional array";

PyObject *fftmod_fft2d(PyObject *self, PyObject *args) {
  PyObject *op1;
  PyArrayObject *ap1, *ret;
  int type, dir, shift, xdim, ydim, len_array, dim;
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
  dtype = PyArray_DescrFromType(type);
  // this presents a PyArrayObject with the correct type and contiguous data
  ap1 = (PyArrayObject *) PyArray_FROM_OTF(op1, type, NPY_IN_ARRAY);
  pytype = ap1->ob_type;
  if (ap1->nd < 2) {
    PyErr_SetString(PyExc_ValueError, "Can't transform a 1-d array");
    Py_XDECREF(ap1);
    return NULL;
  }
  dim = ap1->nd-1;
  xdim = ap1->dimensions[dim];   // n_fe
  ydim = ap1->dimensions[dim-1]; // n_pe
  len_array = 1;
  while(dim >= 0) len_array *= ap1->dimensions[dim--];
  //ret = (PyArrayObject *)PyArray_Empty(ap1->nd, ap1->dimensions, dtype, 0);
  ret = (PyArrayObject *)PyArray_New(pytype, ap1->nd, ap1->dimensions,
				     type, NULL, NULL, 0, 0, (PyObject *) ap1);
  if(type == PyArray_CFLOAT)
    cfft2d((fftwf_complex *)ap1->data, (fftwf_complex *)ret->data,
	   xdim, ydim, len_array, dir, shift);
  else
    zfft2d((fftw_complex *)ap1->data, (fftw_complex *)ret->data,
	   xdim, ydim, len_array, dir, shift);
  Py_DECREF(ap1);
  return PyArray_Return(ret);
}
   
/**************************************************************************
* fft1d                                                                   *
*                                                                         *
* Repeatedly takes a 1D FFT of length len_xform by advancing the zin and  *
* zout pointers len_z/len_xform times. "direction" indices whether the    *
* transform is a forward of reverse FFT. Quadrant shifting a la fftshift  *
* in Matlab is done by modulation in both domains.                        *
**************************************************************************/

void zfft1d(fftw_complex *zin, fftw_complex *zout, 
	    int len_xform, int len_z, int direction, int shift)
{
  fftw_plan FT1D;
  double tog = 1.0;
  int k, nxforms = len_z/len_xform;
  FT1D = fftw_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform,
			    zout, NULL, 1, len_xform, direction,
			    FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  // if shifting, modulate the input
  if(shift) {
    for(k=0; k<len_z; k++) {
      zin[k][0] *= tog;
      zin[k][1] *= tog;
      tog *= -1.0;
    }
  }
  fftw_execute(FT1D);
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  if(shift) {
    for(k=0; k<len_z; k++) {
      if(zin != zout) {
	zin[k][0] *= tog;
	zin[k][1] *= tog;
      }
      if(direction == INVERSE) {
	zout[k][0] *= (tog/ (double) len_xform);
	zout[k][1] *= (tog/ (double) len_xform);
      } else {
	zout[k][0] *= tog;
	zout[k][1] *= tog;
      }
      tog *= -1.0;
    }
  } else {
    if(direction == INVERSE) {
      for(k=0; k<len_z; k++) {
	zout[k][0] *= (1.0/ (double) len_xform);
	zout[k][1] *= (1.0/ (double) len_xform);
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
  float tog = 1.0;
  int k, nxforms = len_z/len_xform;
  FT1D = fftwf_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform,
			     zout, NULL, 1, len_xform, direction,
			     FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  // modulate the input
  if(shift) {
    for(k=0; k<len_z; k++) {
      zin[k][0] *= tog;
      zin[k][1] *= tog;
      tog *= -1.0;
    }
  }
  fftwf_execute(FT1D);
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  if(shift) {
    for(k=0; k<len_z; k++) {
      if(zin != zout) {
	zin[k][0] *= tog;
	zin[k][1] *= tog;
      }
      if(direction == INVERSE) {
	zout[k][0] *= (tog/ (float) len_xform);
	zout[k][1] *= (tog/ (float) len_xform);
      } else {
	zout[k][0] *= tog;
	zout[k][1] *= tog;
      }
      tog *= -1.0;
    }
  } else {
    if(direction == INVERSE) {
      for(k=0; k<len_z; k++) {
	zout[k][0] *= (1.0/ (float) len_xform);
	zout[k][1] *= (1.0/ (float) len_xform);
      }
    }
  }
  fftwf_destroy_plan(FT1D);
  fftwf_cleanup();
}


/**************************************************************************
* fft2d                                                                  *
*                                                                         *
* Transforms the kspace slices to image-space slices. Quadrant "shifting" *
* a la fftshift in Matlab is done by modulation in both domains.          *
**************************************************************************/
void zfft2d(fftw_complex *zin, fftw_complex *zout, int xdim, int ydim,
	    int len_z, int direction, int shift)
{
  double tog = 1.0;
  int k, size_xform = xdim*ydim, nxforms = len_z/(xdim*ydim);
  int dims[2] = {ydim, xdim};
  fftw_plan FT2D;
  
  FT2D = fftw_plan_many_dft(2, dims, nxforms, 
			    zin, NULL, 1, size_xform,
			    zout, NULL, 1, size_xform,
			    direction, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  if(shift) {
    for(k=0; k<len_z; k++) {
      zin[k][0] *= tog;
      zin[k][1] *= tog;
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  }
  fftw_execute(FT2D);
  tog = 1.0;
  if(shift) {
    for(k=0; k<len_z; k++) {
      if(zin != zout) {
	zin[k][0] *= tog;
	zin[k][1] *= tog;
      }
      if(direction==INVERSE) {
	zout[k][0] *= (tog/ (double) size_xform);
	zout[k][1] *= (tog/ (double) size_xform);
      } else {
	zout[k][0] *= tog;
	zout[k][1] *= tog;
      }
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  } else {
    if(direction==INVERSE) {
      for(k=0; k<len_z; k++) {
	zout[k][0] *= (1.0/ (double) size_xform);
	zout[k][1] *= (1.0/ (double) size_xform);
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
  float tog = 1.0;
  int k, size_xform = xdim*ydim, nxforms = len_z/(xdim*ydim);
  int dims[2] = {ydim, xdim};
  fftwf_plan FT2D;
  FT2D = fftwf_plan_many_dft(2, dims, nxforms, 
			     zin, NULL, 1, size_xform,
			     zout, NULL, 1, size_xform,
			     direction, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  if(shift) {
    for(k=0; k<len_z; k++) {
      zin[k][0] *= tog;
      zin[k][1] *= tog;
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  }
  fftwf_execute(FT2D);
  tog = 1.0;
  if(shift) {
    for(k=0; k<len_z; k++) {
      if(zin != zout) {
	zin[k][0] *= tog;
	zin[k][1] *= tog;
      }
      if(direction==INVERSE) {
	zout[k][0] *= (tog/ (float) size_xform);
	zout[k][1] *= (tog/ (float) size_xform);
      } else {
	zout[k][0] *= tog;
	zout[k][1] *= tog;
      }
      if( (k+1)%xdim ) {
	tog *= -1.0;
      }
    }
  } else {
    if(direction==INVERSE) {
      for(k=0; k<len_z; k++) {
	zout[k][0] *= (1.0/ (float) size_xform);
	zout[k][1] *= (1.0/ (float) size_xform);
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
