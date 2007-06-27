#include "recon.h"
#include "util.h"

extern int dgesdd_(char *jobz, int *m, int *n, double *a,
		   int *lda, double *s, double *u, int *ldu,
		   double *vt, int *ldvt, double *work, int *lwork,
		   int *iwork, int *info);

/* returns the maximum element of real array, collects the index
   of the element in max_idx, if it is not a null ptr */
double array_max(double *array, int len, int *max_idx) {
  int k = 0, kmax;
  double max;
  max = array[0];
  kmax = 0;
  for(k=1; k<len; k++) {
    if(array[k] > max) {
      max = array[k];
      kmax = k;
    }
  }
  if( max_idx != NULL ) *max_idx = kmax;
  return max;
}

/* repeatedly takes a 1D FFT of length len_xform by advancing the zin and
   zout pointers len_z/len_xform times. "direction" indices whether the
   transform is a forward of reverse FFT */
void fft1d(fftw_complex *zin, fftw_complex *zout, 
	   int len_xform, int len_z, int direction)
{
  fftw_plan FT1D;
  double tog = 1.0;
  fftw_complex *dp_in, *dp_out;
  int x, k, nxforms = len_z/len_xform;

  FT1D = fftw_plan_dft_1d(len_xform, zin, zout, direction, 
			  FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  
  for(x=0; x<nxforms; x++) {
    dp_in = zin + x*len_xform;
    dp_out = zout + x*len_xform;
/*     FT1D = fftw_plan_dft_1d(len_xform, dp_in, dp_out, direction,  */
/* 			     FFTW_ESTIMATE | FFTW_PRESERVE_INPUT); */
    for(k=0; k<len_xform; k++) {
      dp_in[k][0] *= tog;
      dp_in[k][1] *= tog;
      tog *= -1.0;
    }
    fftw_execute_dft(FT1D, dp_in, dp_out);
/*     fftw_destroy_plan(FT1D);    */
    tog = 1.0;
    for(k=0; k<len_xform; k++) {
      /* undo the modulation in both spaces */
      if(dp_in != dp_out) {
	dp_in[k][0] *= tog;
	dp_in[k][1] *= tog;
      }
      /* FFTW does not normalize on the inverse, so do it here */
      if(direction == INVERSE) {
	dp_out[k][0] *= (tog/ (double) len_xform);
	dp_out[k][1] *= (tog/ (double) len_xform);
      } else {
	dp_out[k][0] *= tog;
	dp_out[k][1] *= tog;
      }
      tog *= -1.0;
    }
  }
  fftw_destroy_plan(FT1D);
  fftw_cleanup();
}

/* returns the variance of the set of points */
double var(double *points, int npts) 
{
  double mean, s;
  int k;
  mean = 0.0;
  s = 0.0;
  for(k=0; k<npts; k++) mean += points[k];
  mean /= ((double) npts);
  for(k=0; k<npts; k++) s += pow(points[k] - mean, 2.0);
  s /= ((double) npts);
  return s;
}

    
/* solves Ax = y for x
   M is the number of rows in A (also the length of y)
   N is the number of cols in A (also the length of x)
   
   via SVD, A = (USV'), then x = (VSU')y
   
   A is an MxN matrix in row-major.. going to feed it into
   LAPACK as if it's a NxM (A') matrix in col-major

   JUST REMEMBER that the arguments (u,s,vt) will actually be (v,s,ut) !!
*/
void dsolve_svd(double *A, double *y, double *x, int M, int N)
{
  char JOBZ = 'S';
  int LDA, LDU, LDVT, LWORK, IWORK, LRWORK, INFO, ns, k;
  double *is, *s, *u, *vt, *work, *vec1, *vec2, cond;

  /* A' is shaped (N,M) */
  LDA = N;
  LDU = N;
  /* number of singular values is min(M,N) */
  ns = MIN(M,N);
  /* u should be (n,ncol) ncol == ns */
  u = (double *) calloc(LDU*ns, sizeof(double));
  LDVT = ns;
  /* vt should be shaped (ldvt, M) */
  vt = (double *) calloc(LDVT*M, sizeof(double));
  LWORK = -1;
  IWORK = 8*ns;
  work = (double *) calloc(1, sizeof(double));
  s = (double *) calloc(ns, sizeof(double));
  dgesdd_(&JOBZ, &N, &M, A, &LDA, s, u, &LDU, vt, &LDVT,
	  work, &LWORK, &IWORK, &INFO);
  LWORK = (int) work[0];
  free(work);
  work = (double *) calloc(LWORK, sizeof(double));
  dgesdd_(&JOBZ, &N, &M, A, &LDA, s, u, &LDU, vt, &LDVT,
	  work, &LWORK, &IWORK, &INFO);
  if(INFO != 0) {
    printf("some error in SVD routine\n");
    exit(1);
  }
  /* U is V, shaped (N,min(N,M))
     V' is U' shaped (min(N,M), M)
     
     now we want to say x = dot(v, dot(inv(s), dot(ut, y)))
     (where v is u and ut is vt)
  */
  is = (double *) calloc((ns*ns), sizeof(double));
  for(k=0; k<ns; k++) is[k + k*ns] = 1.0/s[k];
  
  /* this hold the product dot(ut, y), has length min(N,M), or ns */
  vec1 = (double *) calloc(ns, sizeof(double));
  /* this holds the product dot(inv(s), vec1), has length ns */
  vec2 = (double *) calloc(ns, sizeof(double));
  cblas_dgemv(CblasColMajor, CblasNoTrans, 
	      ns, M, 1.0, vt, ns, 
	      y, 1, 0.0, vec1, 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans,
	      ns, ns, 1.0, is, ns, 
	      vec1, 1, 0.0, vec2, 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, 
	      N, ns, 1.0, u, ns, 
	      vec2, 1, 0.0, x, 1);

  free(vec1);
  free(vec2);
  free(vt);
  free(u);
  free(s);
  free(is);
  free(work);
}

/* Linear Regression routine:
   Arguments y (samples), len, m, b, and res should be valid pointers.
   Arguments x (points of samples) and sigma (variance of samples) may be
   provided, but will be assumed otherwise.
*/
void linReg(double *y, double *x, double *sigma, int len, 
	    double *m, double *b, double *res)
{
  
  int k, local_x = 0, local_sigma = 0;
  double s, sx, sy, sxx, sxy, delta, v, pt;
  
  s = 0.0; sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0;
  
  if(x==NULL) local_x = 1;
  if(sigma==NULL) local_sigma = 1;
      
  for(k=0; k<len; k++) {
    pt = local_x ? (double) k : x[k];
    v = local_sigma ? 1.0 : sigma[k];
    sx += pt/v;
    sy += y[k]/v;
    sxx += (pt * pt)/v;
    sxy += (pt * y[k])/v;
    s += 1.0/v;
  }
  delta = sxx * s - (sx * sx);
  *b = (sxx * sy - sx * sxy)/delta;
  *m = (s * sxy - sx * sy)/delta;
  *res = 0.0;
  for(k=0; k<len; k++) {
    pt = local_x ? (double) k : x[k];
    *res += ABS( y[k] - ((*m)*pt + (*b)) )/(double) len;
  }
}
