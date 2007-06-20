#include "data.h"
#include "recon.h"

/*************************** MEMORY ALLOCATION ***************************/


/* This is the ANSI C (only) version of the Numerical Recipes utility file 
   nrutil.c. It has been modified by DJS to remove the offset=1 stuff.*/


void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

float *vector(long np)
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v=(float *)malloc((size_t) (np*sizeof(float)));
	if (!v) nrerror("allocation failure in vector()");
	return v;
}

int *ivector(long np)
/* allocate an int vector with subscript range v[nl..nh] */
{
	int *v;

	v=(int *)malloc((size_t) (np*sizeof(int)));
	if (!v) nrerror("allocation failure in ivector()");
	return v;
}


double *dvector(long np)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) (np*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector()");
	return v;
}

float **matrix(long nrow, long ncol)

{
        long i;
	float **m;

	/* allocate pointers to rows */
	m=(float **) malloc((size_t)((nrow)*sizeof(float*)));
	if (!m) nrerror("allocation failure 1 in matrix()");


	/* allocate rows and set pointers to them */
	m[0]=(float *) malloc((size_t)((nrow*ncol)*sizeof(float)));
	if (!m[0]) nrerror("allocation failure 2 in matrix()");

	for(i=1;i<=nrow;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

double **dmatrix(long nrow, long ncol)
{
	long i;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((size_t)((nrow)*sizeof(double*)));
	if (!m) nrerror("allocation failure 1 in matrix()");

	/* allocate rows and set pointers to them */
	m[0]=(double *) malloc((size_t)((nrow*ncol)*sizeof(double)));
	if (!m[0]) nrerror("allocation failure 2 in matrix()");

	for(i=1;i<=nrow;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

/* gives a column major matrix for LAPACK use.. 
   it will have to be indexed [x,y] instead of normal [y,x] */
double **dmatrix_colmajor(int nrow, int ncol)
{
  int i;
  double **m;
  m = (double **) malloc(ncol * sizeof(double*));
  if (!m) nrerror("allocation failure in dim 1 in dmatrix_colmajor()");

  m[0] = (double *) malloc((ncol*nrow) * sizeof(double));
  if (!m[0]) nrerror("allocation failure in dim 2 in dmatrix_colmajor()");

  for(i=1; i<ncol; i++) m[i] = m[i-1]+nrow;

  return m;
}

fftw_complex **cmatrix(long nrow, long ncol)
{
	long i;
	fftw_complex **m;

	/* allocate pointers to rows */
	m=(fftw_complex **) malloc((size_t)((nrow)*sizeof(fftw_complex*)));
	if (!m) nrerror("allocation failure 1 in matrix()");

	/* allocate rows and set pointers to them */
	m[0]=(fftw_complex *) fftw_malloc((size_t)((nrow*ncol)*sizeof(fftw_complex)));
	if (!m[0]) nrerror("allocation failure 2 in matrix()");

	for(i=1;i<=nrow;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

/* gives a column major matrix for LAPACK use.. 
   it will have to be indexed [x,y] instead of normal [y,x] */
fftw_complex **cmatrix_colmajor(long nrow, long ncol)
{
  int i;
  fftw_complex **m;
  m = (fftw_complex **) malloc(ncol * sizeof(fftw_complex*));
  if (!m) nrerror("allocation failure in dim 1 in dmatrix_colmajor()");

  m[0] = (fftw_complex *) fftw_malloc((ncol*nrow) * sizeof(fftw_complex));
  if (!m[0]) nrerror("allocation failure in dim 2 in dmatrix_colmajor()");

  for(i=1; i<ncol; i++) m[i] = m[i-1]+nrow;

  return m;
}

float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
	long newrl, long newcl)
/* point a submatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
{
	long i,j,nrow=oldrh-oldrl+1,ncol=oldcl-newcl;
	float **m;

	/* allocate array of pointers to rows */
	m=(float **) malloc((size_t) ((nrow)*sizeof(float*)));
	if (!m) nrerror("allocation failure in submatrix()");
	m -= newrl;

	/* set pointers to rows */
	for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

fftw_complex ***c3tensor_alloc(long nsl, long nrow, long ncol)
/* allocate a fftw_complex 3tensor with dimensions (nsl, nrow, ncol) */
{
        long i,j;
	fftw_complex ***t;

	/* allocate pointers to pointers to rows */
	t=(fftw_complex ***) malloc((size_t)((nsl)*sizeof(fftw_complex**)));
	if (!t) nrerror("allocation failure 1 in c3tensor()");

	/* allocate pointers to rows and set pointers to them */
	t[0]=(fftw_complex **) malloc((size_t)((nsl*nrow)*sizeof(fftw_complex*)));
	if (!t[0]) nrerror("allocation failure 2 in c3tensor()");

	/* allocate rows and set pointers to them */
	t[0][0]=(fftw_complex *) fftw_malloc((size_t)((nsl*nrow*ncol)*sizeof(fftw_complex)));
	if (!t[0][0]) nrerror("allocation failure 3 in c3tensor()");

	for(j=1;j<nrow;j++) t[0][j]=t[0][j-1]+ncol;
	for(i=1;i<nsl;i++) {
		t[i]=t[i-1]+nrow;
		t[i][0]=t[i-1][0]+nrow*ncol;
		for(j=1;j<nrow;j++) t[i][j]=t[i][j-1]+ncol;
	}

	/* return pointer to array of pointers to rows */
	return t;
}

double ***d3tensor_alloc(long nsl, long nrow, long ncol)
/* allocate a double type 3tensor with dimensions (nsl, nrow, ncol) */
{
        long i,j;
	double ***t;

	/* allocate pointers to pointers to rows */
	t=(double ***) malloc((size_t)((nsl)*sizeof(double**)));
	if (!t) nrerror("allocation failure 1 in d3tensor()");

	/* allocate pointers to rows and set pointers to them */
	t[0]=(double **) malloc((size_t)((nsl*nrow)*sizeof(double*)));
	if (!t[0]) nrerror("allocation failure 2 in d3tensor()");

	/* allocate rows and set pointers to them */
	//t[0][0]=(double *) malloc((size_t)((nsl*nrow*ncol)*sizeof(double)));
	t[0][0]=(double *) calloc((size_t)(nsl*nrow*ncol), sizeof(double));
	if (!t[0][0]) nrerror("allocation failure 3 in d3tensor()");

	for(j=1;j<nrow;j++) t[0][j]=t[0][j-1]+ncol;
	for(i=1;i<nsl;i++) {
		t[i]=t[i-1]+nrow;
		t[i][0]=t[i-1][0]+nrow*ncol;
		for(j=1;j<nrow;j++) t[i][j]=t[i][j-1]+ncol;
	}

	/* return pointer to array of pointers to rows */
	return t;
}


fftw_complex ****c4tensor_alloc(long nvol, long nsl, long nrow, long ncol)
/* allocate a fftw_complex 4D array */
{
  long k, l, m;
  fftw_complex ****t;
  t = (fftw_complex ****) malloc( nvol * sizeof(fftw_complex ***) );
  if (!t) nrerror("dimension 0 allocation failed c4tensor()");

  t[0] = (fftw_complex ***) malloc( (nvol*nsl) * sizeof(fftw_complex **));
  if (!t[0]) nrerror("dimensions 1 allocation failed c4tensor()");

  t[0][0] = (fftw_complex **) malloc( (nvol*nsl*nrow) * sizeof(fftw_complex *));
  if (!t[0][0]) nrerror("dimension 2 allocation failed c4tensor()");

  t[0][0][0] = (fftw_complex *) fftw_malloc( (nvol*nsl*nrow*ncol) * sizeof(fftw_complex));
  if(!t[0][0][0]) nrerror("dimension 3 allocation failed c4tensor()");

  /* arange 0th level pointers:  t[0][0...nsl-1][0...nrow-1]  */
  for(m=1; m<nrow; m++) t[0][0][m] = t[0][0][m-1] + ncol;
  for(l=1; l<nsl; l++) {
    t[0][l] = t[0][l-1] + nrow;
    t[0][l][0] = t[0][l-1][0] + nrow*ncol;
    for(m=1; m<nrow; m++) t[0][l][m] = t[0][l][m-1] + ncol;
  }

  /* now iterate through t[1..nvol-1][...][...] */
  for(k=1 ; k<nvol; k++) {
    t[k] = t[k-1] + nsl;
    t[k][0] = t[k-1][0] + nsl*nrow;
    t[k][0][0] = t[k-1][0][0] + nsl*nrow*ncol;
    for(m=1; m<nrow; m++) t[k][0][m] = t[k][0][m-1] + ncol;
    for(l=1; l<nsl; l++) {
      t[k][l] = t[k][l-1] + nrow;
      t[k][l][0] = t[k][l-1][0] + nrow*ncol;
      for(m=1; m<nrow; m++) t[k][l][m] = t[k][l][m-1] + ncol;
    }
  }
  return t;
}

double ****d4tensor_alloc(long nvol, long nsl, long nrow, long ncol)
/* allocate a double 4D array */
{
  long k, l, m;
  double ****t;
  t = (double ****) malloc( nvol * sizeof(double ***) );
  if (!t) nrerror("dimension 0 allocation failed c4tensor()");

  t[0] = (double ***) malloc( (nvol*nsl) * sizeof(double **));
  if (!t[0]) nrerror("dimensions 1 allocation failed c4tensor()");

  t[0][0] = (double **) malloc( (nvol*nsl*nrow) * sizeof(double *));
  if (!t[0][0]) nrerror("dimension 2 allocation failed c4tensor()");

  t[0][0][0] = (double *) malloc( (nvol*nsl*nrow*ncol) * sizeof(double));
  if(!t[0][0][0]) nrerror("dimension 3 allocation failed c4tensor()");

  /* arange 0th level pointers:  t[0][0...nsl-1][0...nrow-1]  */
  for(m=1; m<nrow; m++) t[0][0][m] = t[0][0][m-1] + ncol;
  for(l=1; l<nsl; l++) {
    t[0][l] = t[0][l-1] + nrow;
    t[0][l][0] = t[0][l-1][0] + nrow*ncol;
    for(m=1; m<nrow; m++) t[0][l][m] = t[0][l][m-1] + ncol;
  }

  /* now iterate through t[1..nvol-1][...][...] */
  for(k=1 ; k<nvol; k++) {
    t[k] = t[k-1] + nsl;
    t[k][0] = t[k-1][0] + nsl*nrow;
    t[k][0][0] = t[k-1][0][0] + nsl*nrow*ncol;
    for(m=1; m<nrow; m++) t[k][0][m] = t[k][0][m-1] + ncol;
    for(l=1; l<nsl; l++) {
      t[k][l] = t[k][l-1] + nrow;
      t[k][l][0] = t[k][l-1][0] + nrow*ncol;
      for(m=1; m<nrow; m++) t[k][l][m] = t[k][l][m-1] + ncol;
    }
  }
  return t;
}


fftw_complex *zarray(long dsize)
/* allocate a complex array linear in memory, required by fftw */
{
	fftw_complex *t;
	t = (fftw_complex *) fftw_malloc(dsize * sizeof(fftw_complex));

	return t;
}


void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
	free((FREE_ARG) (v+nl));
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
	free((FREE_ARG) (v+nl));
}


void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
	free((FREE_ARG) (v+nl));
}

void free_matrix(float **m)
/* free a float matrix allocated by matrix() */
{
	free((FREE_ARG) m[0]);
	free((FREE_ARG) m);
}

void free_dmatrix(double **m)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG) m[0]);
	free((FREE_ARG) m);
}

void free_cmatrix(fftw_complex **m)
{
  fftw_free(m[0]);
  fftw_free(m);
}

void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a submatrix allocated by submatrix() */
{
	free((FREE_ARG) (b+nrl));
}

void free_c3tensor(fftw_complex ***t)
/* free a float c3tensor allocated by c3tensor() */
{
	fftw_free((FREE_ARG) t[0][0]);
	free((FREE_ARG) t[0]);
	free((FREE_ARG) t);
}

void free_d3tensor(double ***t)
/* free a float d3tensor allocated by d3tensor() */
{
	free((FREE_ARG) t[0][0]);
	free((FREE_ARG) t[0]);
	free((FREE_ARG) t);
}


void free_c4tensor(fftw_complex ****t)
{
  fftw_free((FREE_ARG) t[0][0][0]);
  free((FREE_ARG) t[0][0]);
  free((FREE_ARG) t[0]);
  free((FREE_ARG) t);
}

void free_d4tensor(double ****t)
{
  free((FREE_ARG) t[0][0][0]);
  free((FREE_ARG) t[0][0]);
  free((FREE_ARG) t[0]);
  free((FREE_ARG) t);
}


void free_zarray(fftw_complex *t)
{
  free(t);
}


double *mag(double *r, const fftw_complex *z, long N) 
{
  int k;
  double re, im;
  for(k=0; k<N; k++) {
    re = z[k][0];
    im = z[k][1];
    r[k] = sqrt(re*re + im*im);
  }
}

double *angle(double *r, const fftw_complex *z, long N)
{
  int k;
  for(k=0; k<N; k++) r[k] = atan2(z[k][1], z[k][0]);
}

double *real(double *r, const fftw_complex *z, long N)
{
  int k;
  for(k=0; k<N; k++) r[k] = z[k][0];
}

double *imag(double *r, const fftw_complex *z, long N)
{
  int k;
  for(k=0; k<N; k++) r[k] = z[k][1];
}
