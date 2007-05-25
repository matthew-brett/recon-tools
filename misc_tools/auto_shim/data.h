#include <fftw3.h>
#define FREE_ARG char*


float **matrix(long nrl, long nrh, long ncl, long nch);
fftw_complex ***f3tensor(long nr, long nc, long nd);
fftw_complex *zarray(long dsize);
