#include <fftw3.h>
#define FREE_ARG char*


void nrerror(char error_text[]);
float *vector(long np);
int *ivector(long np);
double *dvector(long np);
float **matrix(long nrow, long ncol);
double **dmatrix(long nrow, long ncol);
double **dmatrix_colmajor(int nrow, int ncol);
float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
		  long newrl, long newcl);
void free_vector(float *v, long nl, long nh);
void free_ivector(int *v, long nl, long nh);
void free_dvector(double *v, long nl, long nh);
void free_matrix(float **m);
void free_dmatrix(double **m);
void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch);
void free_d3tensor(double ***t);
void free_c3tensor(fftw_complex ***t);
void free_c4tensor(fftw_complex ****t);
void free_d4tensor(double ****t);
void free_zarray(fftw_complex *t);

double ***d3tensor_alloc(long nsl, long nrow, long ncol);
fftw_complex ***c3tensor_alloc(long nsl, long nrow, long ncol);
fftw_complex ****c4tensor_alloc(long nvol, long nsl, long nrow, long ncol);
double ****d4tensor_alloc(long nvol, long nsl, long nrow, long ncol);

fftw_complex *zarray(long dsize);

/* Definitions for typical complex->real data conversions */
double *mag(double *r, const fftw_complex *z, long N);
double *angle(double *r, const fftw_complex *z, long N);
double *real(double *r, const fftw_complex *z, long N);
double *imag(double *r, const fftw_complex *z, long N);
