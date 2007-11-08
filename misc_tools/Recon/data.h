#include <fftw3.h>
#define FREE_ARG char*


void nrerror(char error_text[]);
float *vector(int np);
int *ivector(int np);
double *dvector(int np);
float **matrix(int nrow, int ncol);
double **dmatrix(int nrow, int ncol);
double **dmatrix_colmajor(int nrow, int ncol);
fftw_complex **cmatrix(int nrow, int ncol);
fftw_complex **cmatrix_colmajor(int nrow, int ncol);
float **submatrix(float **a, int oldrl, int oldrh, int oldcl, int oldch,
		  int newrl, int newcl);
void free_vector(float *v, int nl, int nh);
void free_ivector(int *v, int nl, int nh);
void free_dvector(double *v, int nl, int nh);
void free_matrix(float **m);
void free_dmatrix(double **m);
void free_cmatrix(fftw_complex **m);
void free_submatrix(float **b, int nrl, int nrh, int ncl, int nch);
void free_d3tensor(double ***t);
void free_c3tensor(fftw_complex ***t);
void free_c4tensor(fftw_complex ****t);
void free_d4tensor(double ****t);
void free_zarray(fftw_complex *t);

double ***d3tensor_alloc(int nsl, int nrow, int ncol);
fftw_complex ***c3tensor_alloc(int nsl, int nrow, int ncol);
fftw_complex ****c4tensor_alloc(int nvol, int nsl, int nrow, int ncol);
double ****d4tensor_alloc(int nvol, int nsl, int nrow, int ncol);

fftw_complex *zarray(int dsize);

/* Definitions for typical complex->real data conversions */
void mag(double *r, const fftw_complex *z, int N);
void angle(double *r, const fftw_complex *z, int N);
void real(double *r, const fftw_complex *z, int N);
void imag(double *r, const fftw_complex *z, int N);
