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
#include "Munther_2D_unwrap.h"

static char doc_Unwrap2D[] = "Performs 2D phase unwrapping on an ndarray";

PyObject *punwrap2D_Unwrap2D(PyObject *self, PyObject *args) {
  PyObject *op1, *op2;
  PyArrayObject *ap1, *ap2, *ret;
  int typenum_phs, typenum_mask, nd;
  int dimensions[2];
  PyArray_Descr *dtype_phs, *dtype_mask;
  PyTypeObject *subtype;

  if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) {
    printf("couldn't parse any args\n");
    return NULL;
  }
  if(op1==NULL || op2==NULL) {
    printf("args not read correctly\n");
    return NULL;
  }
  typenum_phs = PyArray_ObjectType(op1,0);
  typenum_mask = PyArray_ObjectType(op2,0);
  if(typenum_phs != PyArray_FLOAT) {
    PyErr_SetString(PyExc_TypeError, "Currently I can only handle single-precision floating point numbers");
    return NULL;
  }
  if(typenum_mask != PyArray_UBYTE) {
    PyErr_SetString(PyExc_TypeError, "The mask should be type uint8");
    return NULL;
  }

  dtype_phs = PyArray_DescrFromType(typenum_phs);
  dtype_mask = PyArray_DescrFromType(typenum_mask);
  ap1 = (PyArrayObject *)PyArray_FROM_OTF(op1, typenum_phs, NPY_IN_ARRAY);
  ap2 = (PyArrayObject *)PyArray_FROM_OTF(op2, typenum_mask, NPY_IN_ARRAY);
  subtype = ap1->ob_type;
  nd = ap1->nd;
  dimensions[0] = ap1->dimensions[0]; dimensions[1] = ap1->dimensions[1];

  if(ap1->nd != 2) {
    PyErr_SetString(PyExc_ValueError, "I can only unwrap 2D arrays");
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    return NULL;
  }

  ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
				     typenum_phs, NULL, NULL, 0, 0,
				     (PyObject *) ap1);

  phase_unwrap_2D((float *) ap1->data, (float *) ret->data, (BYTE *) ap2->data,
		  dimensions[0], dimensions[1]);

/*   doUnwrap((float *) ap1->data, (float *) ret->data, (long) dimensions[0], (long) dimensions[1]); */
  
  Py_DECREF(ap1);
  Py_DECREF(ap2);
  return PyArray_Return(ret);
    
}

static struct PyMethodDef punwrap2D_module_methods[] = {
  {"Unwrap2D",	(PyCFunction)punwrap2D_Unwrap2D, 1, doc_Unwrap2D},
  {NULL, NULL, 0}
};

DL_EXPORT(void) init_punwrap2D(void) {
  PyObject *m;
  m = Py_InitModule3("_punwrap2D", punwrap2D_module_methods, "c library for phase unwrapping in python");
  import_array();
}
