#include <stdio.h>

#include "Python.h"
#include "numpy/noprefix.h"
#include "UnwrapMain.h"

static char doc_Unwrap3D[] = "Performs 3D phase unwrapping on an ndarray";

PyObject *punwrap3D_Unwrap3D(PyObject *self, PyObject *args) {
  PyObject *op1;
  PyArrayObject *ap1, *ret;
  int typenum_phs;
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
  dtype = PyArray_DescrFromType(typenum_phs);
  ap1 = (PyArrayObject *)PyArray_FROM_OTF(op1, typenum_phs, NPY_IN_ARRAY);
  subtype = ap1->ob_type;
  if(ap1->nd < 3) {
    PyErr_SetString(PyExc_ValueError, "I can only unwrap 3D arrays");
    Py_XDECREF(ap1);
    return NULL;
  }

  ret = (PyArrayObject *)PyArray_New(subtype, ap1->nd, ap1->dimensions,
				     typenum_phs, NULL, NULL, 0, 0,
				     (PyObject *) ap1);
  unwrap_phs((float *) ap1->data, (float *) ret->data, 
	     (int) ap1->dimensions[0], 
	     (int) ap1->dimensions[1], 
	     (int) ap1->dimensions[2]);
  
  Py_DECREF(ap1);
  return PyArray_Return(ret);
    
}

static struct PyMethodDef punwrap3D_module_methods[] = {
  {"Unwrap3D",	(PyCFunction)punwrap3D_Unwrap3D, 1, doc_Unwrap3D},
  {NULL, NULL, 0}
};

DL_EXPORT(void) init_punwrap3D(void) {
  PyObject *m;
  m = Py_InitModule3("_punwrap3D", punwrap3D_module_methods, "c library for phase unwrapping in python");
  import_array();
}
