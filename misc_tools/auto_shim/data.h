#include <fftw3.h>
#define FREE_ARG char*


void nrerror(char error_text[]);
float *vector(long nl, long nh);
int *ivector(long nl, long nh);
double *dvector(long nl, long nh);
float **matrix(long nrl, long nrh, long ncl, long nch);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
		  long newrl, long newcl);
void free_vector(float *v, long nl, long nh);
void free_ivector(int *v, long nl, long nh);
void free_dvector(double *v, long nl, long nh);
void free_matrix(float **m, long nrl, long nrh, long ncl, long nch);
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);
void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch);
void free_f3tensor(float ***t, long nr, long nc, long nd);
void free_zarray(fftw_complex *t);

fftw_complex ***f3tensor(long nr, long nc, long nd);
fftw_complex *zarray(long dsize);
