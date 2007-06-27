#include <fftw3.h>
#include <vecLib/cblas.h>

enum ftdirections {FORWARD=-1, INVERSE=+1};


double array_max(double *array, int len, int *max_idx);
void reverse_fe(fftw_complex *z, int n_fe, int len_z);
void fft1d(fftw_complex *zin, fftw_complex *zout, 
	   int len_xform, int len_z, int direction);
double var(double *points, int npts);
void dsolve_svd(double *A, double *y, double *x, int M, int N);
//void zsolve_svd(fftw_complex *A, fftw_complex *y, fftw_complex *x, int M, int N);
void linReg(double *y, double *x, double *sigma, int len, 
	    double *m, double *b, double *res);

