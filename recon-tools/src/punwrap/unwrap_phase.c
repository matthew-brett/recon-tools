/*
 *  unwrap_phase.c
 *  
 *
 *  Created by miket on 6/13/06.
 *  Copyright 2006 Mike Trumpis.
 *
 */
#include <stdio.h>

#include "Python.h"
#include "numpy/noprefix.h"
#include "snaphu_unwrap.h"

static char doc_Unwrap[] = "Performs 2D phase unwrapping on a Numeric array via the LP-norm method";

PyObject *punwrap_Unwrap(PyObject *self, PyObject *args) {
  PyObject *op1;
  PyArrayObject *ap1, *ret;
  int typenum_phs, k, j;
  int nlines, nd;
  intp dimensions[2];
  PyArray_Descr *dtype;
  PyTypeObject *subtype;
    
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
    return NULL;
  }
  /*ap1 = (PyArrayObject *)PyArray_ContiguousFromObject(op1, typenum_phs, 0, 0);*/
  dtype = PyArray_DescrFromType(typenum_phs);
  ap1 = (PyArrayObject *)PyArray_FromAny(op1, dtype, 0, 0, ALIGNED, NULL);
  subtype = ap1->ob_type;
  nd = ap1->nd;
  dimensions[0] = ap1->dimensions[0]; dimensions[1] = ap1->dimensions[1];

  if(ap1->nd != 2) {
    PyErr_SetString(PyExc_ValueError, "I can only unwrap 2D arrays");
    Py_XDECREF(ap1);
    return NULL;
  }
  /*ret = (PyArrayObject *)PyArray_FromDims(ap1->nd, ap1->dimensions, typenum_phs);*/
  ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
				     typenum_phs, NULL, NULL, 0, 0,
				     (PyObject *) ap1);
  doUnwrap((float *) ap1->data, (float *) ret->data, (long) dimensions[0], (long) dimensions[1]);
  
  Py_DECREF(ap1);
  return PyArray_Return(ret);
    
}

static struct PyMethodDef punwrap_module_methods[] = {
  {"Unwrap",	(PyCFunction)punwrap_Unwrap, 1, doc_Unwrap},
  {NULL, NULL, 0}
};

DL_EXPORT(void) init_punwrap(void) {
  PyObject *m;
  m = Py_InitModule3("_punwrap", punwrap_module_methods, "c library for phase unwrapping in python");
  import_array();
}
