#include "data.h"
#include "auto_shim.h"

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

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v=(float *)malloc((size_t) ((nh-nl+1)*sizeof(float)));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl;
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
	int *v;

	v=(int *)malloc((size_t) ((nh-nl+1)*sizeof(int)));
	if (!v) nrerror("allocation failure in ivector()");
	return v-nl;
}


double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1)*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector()");
	return v-nl;
}

float **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	float **m;

	/* allocate pointers to rows */
	m=(float **) malloc((size_t)((nrow)*sizeof(float*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(float *) malloc((size_t)((nrow*ncol)*sizeof(float)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((size_t)((nrow)*sizeof(double*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((size_t)((nrow*ncol)*sizeof(double)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
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

fftw_complex ***f3tensor(long nr, long nc, long nd)
/* allocate a fftw_complex 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	long i,j,nrow=nr,ncol=nc,ndep=nd;
	fftw_complex ***t;

	/* allocate pointers to pointers to rows */
	t=(fftw_complex ***) malloc((size_t)((nrow)*sizeof(fftw_complex**)));
	if (!t) nrerror("allocation failure 1 in f3tensor()");

	/* allocate pointers to rows and set pointers to them */
	t[0]=(fftw_complex **) malloc((size_t)((nrow*ncol)*sizeof(fftw_complex*)));
	if (!t[0]) nrerror("allocation failure 2 in f3tensor()");

	/* allocate rows and set pointers to them */
	t[0][0]=(fftw_complex *) malloc((size_t)((nrow*ncol*ndep)*sizeof(fftw_complex)));
	if (!t[0][0]) nrerror("allocation failure 3 in f3tensor()");

	for(j=1;j<ncol;j++) t[0][j]=t[0][j-1]+ndep;
	for(i=1;i<nrow;i++) {
		t[i]=t[i-1]+ncol;
		t[i][0]=t[i-1][0]+ncol*ndep;
		for(j=1;j<ncol;j++) t[i][j]=t[i][j-1]+ndep;
	}

	/* return pointer to array of pointers to rows */
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

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
	free((FREE_ARG) (m[nrl]+ncl));
	free((FREE_ARG) (m+nrl));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG) (m[nrl]+ncl));
	free((FREE_ARG) (m+nrl));
}


void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a submatrix allocated by submatrix() */
{
	free((FREE_ARG) (b+nrl));
}

void free_f3tensor(float ***t, long nr, long nc, long nd)
/* free a float f3tensor allocated by f3tensor() */
{
	free((FREE_ARG) (t[nr-1][nc-1]+nd-1));
	free((FREE_ARG) (t[nr-1]+nc-1));
	free((FREE_ARG) (t+nr-1));
}

void free_zarray(fftw_complex *t)
{
  free(t);
}

