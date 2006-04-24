#include "lpnorm.h"
#include <string.h>
#include "Python.h"
#include "Numeric/arrayobject.h"

#define BORDER 0x20

static void pullSubGrid(float *dest, const float *src, int gx, int gy, int sgx, int sgy);

static char doc_lpUnwrap[] = "Performs 2D phase unwrapping on a 2D Numeric array, via the LP-norm method\nData should come in/out normalized from (0,1)";

PyObject *punwrap_lpUnwrap(PyObject *self, PyObject *args) {

  PyObject *op1;
  PyArrayObject *ap1, *ret;
  int xsize, ysize, DCT_xsize, DCT_ysize;
  int typenum = 34;
  int nd, dimensions[2];
  int i,j;
  float *dct_phase, *dct_soln, *soln, *rarray, *zarray, *parray, *dxwts, *dywts;
  
  float *qual_map;          // will be zeros
  unsigned char *bitflags;	// will be zeros
  
  if(!PyArg_ParseTuple(args, "O", &op1)) {
    printf("couldn't parse any args\n");
    return NULL;
  }
  if(op1==NULL) {
    printf("op1 still null!\n");
    return NULL;
  }
//  printf("parsed 1 object: op1 is at memaddr %x\n",op1);
  typenum = PyArray_ObjectType(op1, 0);
//  printf("typecode is %d\n",typenum);
//  printf("data type is 32bit Floating point: %d\n", typenum==PyArray_FLOAT);
  if(typenum != PyArray_FLOAT) {
      PyErr_SetString(PyExc_TypeError, "Currently I can only handle single-precision floating point numbers");
//      Py_XDECREF(ap1);
      return NULL;
  }
  ap1 = (PyArrayObject *)PyArray_ContiguousFromObject(op1, typenum, 0, 0);
  
//  printf("number of dims: %d\n", ap1->nd);
  if(ap1->nd != 2) {
    PyErr_SetString(PyExc_ValueError, "I can only unwrap 2D arrays");
    Py_XDECREF(ap1);
    return NULL;
  }

//  printf("nrows: %d, ncols: %d\n", ap1->dimensions[0], ap1->dimensions[1]);
  ysize = ap1->dimensions[0];
  dimensions[0] = ysize;
  xsize = ap1->dimensions[1];
  dimensions[1] = xsize;
  nd = ap1->nd;
  
  for(DCT_ysize = 1; DCT_ysize+1 < ysize; DCT_ysize*=2)
    ;
  for(DCT_xsize = 1; DCT_xsize+1 < xsize; DCT_xsize*=2)
    ;
  DCT_ysize++; DCT_xsize++;
  dct_phase = (float *) calloc(DCT_ysize*DCT_xsize, sizeof(float));
  dct_soln = (float *) calloc(DCT_ysize*DCT_xsize, sizeof(float));
  rarray = (float *) calloc(DCT_ysize*DCT_xsize, sizeof(float));
  zarray = (float *) calloc(DCT_ysize*DCT_xsize, sizeof(float));
  parray = (float *) calloc(DCT_ysize*DCT_xsize, sizeof(float));
  dxwts = (float *) calloc(DCT_ysize*DCT_xsize, sizeof(float));
  dywts = (float *) calloc(DCT_ysize*DCT_xsize, sizeof(float));
  qual_map = (float *) calloc(DCT_ysize*DCT_xsize, sizeof(float));
  bitflags = (unsigned char *) calloc(DCT_ysize*DCT_xsize, sizeof(char));
  
  memmove(dct_phase, ap1->data, ysize*xsize*sizeof(float));
  
   /* embed arrays in possibly larger FFT/DCT arrays */

  for (j=DCT_ysize-1; j>=0; j--) {
    for (i=DCT_xsize-1; i>=0; i--) {
        if (i<xsize && j<ysize) {
            dct_phase[j*DCT_xsize + i] = dct_phase[j*xsize + i];
            qual_map[j*DCT_xsize + i] = 1.0;
        }
        else {
            dct_phase[j*DCT_xsize + i] = 0.0;
            bitflags[j*DCT_xsize + i] = BORDER;
        }
    }
  }
//  printf("ysize for DCT: %d, xsize for DCT: %d\n", DCT_ysize, DCT_xsize);

  int num_iter = 10; int pcg_iter = 20; double e0 = 0.001;
  //printf("ysize,xsize=(%d,%d)\n", ysize, xsize);

  LpNormUnwrap(dct_soln, dct_phase, dxwts, dywts, bitflags, qual_map,
               rarray, zarray, parray, num_iter, pcg_iter, e0, DCT_xsize, DCT_ysize);
  
  
/*  printf("2nd row of soln:\n[");
  for(i=0; i<xsize; i++) printf("%f, ",dct_soln[DCT_xsize+i]);
  printf("]\n");
*/  
  ret = (PyArrayObject *)PyArray_FromDims(nd, dimensions, typenum);
  
  pullSubGrid((float *) ret->data, dct_soln, DCT_xsize, DCT_ysize, dimensions[1], dimensions[0]);
  
  //memmove(ret->data, (char *) dct_soln, xsize*ysize*sizeof(float));
  
  free(dct_phase);
  free(dct_soln);
  free(rarray);
  free(zarray);
  free(parray);
  free(dxwts);
  free(dywts);
  free(qual_map);
  free(bitflags);
   
  Py_DECREF(ap1);
  return PyArray_Return(ret);
  
}

static void pullSubGrid(float *dest, const float *src, int gx, int gy, int sgx, int sgy) {
    // pulls sgx*sgy floats from the top-left corner of src matrix
    int k;
    char *dptr, *sptr;
    dptr = (char *) dest;
    sptr = (char *) src;
    for(k=0; k<sgy; k++) {
        memmove(dptr, sptr, sgx*sizeof(float));
        sptr += gx*sizeof(float*);
        dptr += sgx*sizeof(float*);
    }
}

static struct PyMethodDef punwrap_module_methods[] = {
  {"lpUnwrap",	(PyCFunction)punwrap_lpUnwrap, 1, doc_lpUnwrap},
  {NULL, NULL, 0}
};

DL_EXPORT(void) init_punwrap(void) {
  PyObject *m;
  
  m = Py_InitModule3("_punwrap", punwrap_module_methods, "c library for phase unwrapping in python");
  import_array();
}
