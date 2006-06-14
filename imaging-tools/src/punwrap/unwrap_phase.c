/*
 *  untitled.c
 *  
 *
 *  Created by miket on 6/13/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */
#include <stdio.h>

#include "Python.h"
#include "Numeric/arrayobject.h"
#include "snaphu_unwrap.h"

static char doc_lpUnwrap[] = "Performs 2D phase unwrapping on a 2D Numeric array, via the LP-norm method";

PyObject *punwrap_lpUnwrap(PyObject *self, PyObject *args) {
  PyObject *op1;
  PyArrayObject *ap1, *ret;
  int typenum_phs, k, j;
  int nlines, nd, dimensions[2];
  
  if(!PyArg_ParseTuple(args, "O", &op1)) {
    printf("couldn't parse any args\n");
    return NULL;
  }
  if(op1==NULL) {
    printf("op1 not read correctly\n");
    return NULL;
  }
  typenum_phs = PyArray_ObjectType(op1,0);
  if(typenum_phs != PyArray_FLOAT) {
    PyErr_SetString(PyExc_TypeError, "Currently I can only handle single-precision floating point numbers");
//      Py_XDECREF(ap1);
    return NULL;
  }
  
  ap1 = (PyArrayObject *)PyArray_ContiguousFromObject(op1, typenum_phs, 0, 0);
  
  if(ap1->nd != 2) {
    PyErr_SetString(PyExc_ValueError, "I can only unwrap 2D arrays");
    Py_XDECREF(ap1);
    return NULL;
  }

  ret = (PyArrayObject *)PyArray_FromDims(ap1->nd, ap1->dimensions, typenum_phs);
  doUnwrap((float *) ap1->data, (float *) ret->data, (long) ap1->dimensions[0], (long) ap1->dimensions[1]);
  
  Py_DECREF(ap1);
  return PyArray_Return(ret);
    
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