# now visualization is a module ...
import numpy as N
def iscomplex(a): return a.dtype.kind is 'c'

# Transforms for viewing different aspects of complex data
def ident_xform(data): return data
def abs_xform(data): return N.abs(data)
def phs_xform(data): return N.angle(data)
def real_xform(data): return data.real
def imag_xform(data): return data.imag

