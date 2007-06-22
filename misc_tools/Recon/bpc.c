#include "recon.h"
#include "ops.h"
#include "data.h"

extern int dgesdd_(char *jobz, int *m, int *n, double *a,
		       int *lda, double *s, double *u, int *ldu,
		       double *vt, int *ldvt, double *work, int *lwork,
		       int *iwork, int *info);

void bal_phs_corr(image_struct *image, op_struct op)
{
  fftw_complex ***invref, ***invref1, ***invref2, *ir1, *ir2, *ir, ***pcor_vol;
  double ***phsvol, ***sigma, ***q1_mask, **A, *col, *soln;
  double re1, re2, im1, im2, zarg, mask_tol, mask_tolgrowth;
  int k, l, m, n, n_fe, n_pe, n_slice, dsize, nrows, rc;
  FILE *fp;

  n_fe = image->n_fe;
  n_pe = image->n_pe;
  n_slice = image->n_slice;
  dsize = n_slice * n_pe * n_fe;
  /* memory allocation */
  invref1 = c3tensor_alloc(n_slice, n_pe, n_fe);
  invref2 = c3tensor_alloc(n_slice, n_pe, n_fe);
  invref = c3tensor_alloc(n_slice, n_pe, n_fe);  
  phsvol = d3tensor_alloc(n_slice, n_pe, n_fe);

  sigma = d3tensor_alloc(n_slice, n_pe, n_fe);
  q1_mask = d3tensor_alloc(n_slice, n_pe, n_fe);

  pcor_vol = c3tensor_alloc(n_slice, n_pe, n_fe);
  // make sure this one's contiguous
  soln = (double *) calloc(3, sizeof(double));

  /* done with memory allocation */

  /* mutliply the 1d-ifft of ref1 by the complex conjugate of the 1d-ifft
     of the REVERSED ref2 */
  fft1d(**image->ref1, **invref1, n_fe, dsize, INVERSE);
  reverse_fe(**image->ref2, n_fe, dsize);
  fft1d(**image->ref2, **invref2, n_fe, dsize, INVERSE);

  ir1 = **invref1;
  ir2 = **invref2;
  ir = **invref;
  for(k=0; k<dsize; k++) {
    re1 = ir1[k][0];
    re2 = ir2[k][0];
    im1 = ir1[k][1];
    im2 = ir2[k][1];
    ir[k][0] = re1*re2 + im1*im2;
    ir[k][1] = -re1*im2 + re2*im1;
  }
  
  unwrap_ref_volume(**phsvol, invref, n_slice, n_pe, n_fe, 0, 0);
  /* can get rid of invref stuff now */
  free_c3tensor(invref1);
  free_c3tensor(invref2);
  free_c3tensor(invref);  

  /* Should set up variance and mask arrays too.. */
  /* The variance is wrt even and odd read-outs.. ie: what's the variance
     of a point across similarly polarized read-outs. */

  col = (double *) malloc(n_pe/2 * sizeof(double));
  for(k=0; k<n_slice; k++) {
    for(m=0; m<n_fe; m++) {

      /* get variance of evens */
      for(l=0; l<n_pe/2; l++) col[l] = phsvol[k][2*l][m];
      sigma[k][0][m] = var(col, n_pe/2);
      /* get variance of odds */
      for(l=0; l<n_pe/2; l++) col[l] = phsvol[k][2*l+1][m];
      sigma[k][1][m] = var(col, n_pe/2);

      /* it's nice to repeat the variances along the y dimension, so that
	 the sigma volume has the same shape as the data */
      for(l=1; l<n_pe/2; l++) {
	sigma[k][2*l][m] = sigma[k][0][m];
	sigma[k][2*l+1][m] = sigma[k][1][m];
      }
    }
  }
  /* free col to reuse soon */
  free(col);
  /* MASK BY PSS HERE ONE DAY! */
  mask_tol = 1.25;
  mask_tolgrowth = 1.25;
  for(k=0; k<n_slice; k++) {
    for(l=0; l<n_pe; l++) {
      for(m=0; m<n_fe; m++) q1_mask[k][l][m] = 1.0;
      maskbyfit(phsvol[k][l], sigma[k][l], q1_mask[k][l],
		mask_tol, mask_tolgrowth, n_fe);
    }
  }
  /* For each mu in n_pe, solve for the planar fit of the surface. */
  for(l=0; l<n_pe; l++) {
    
    /* The # of rows in A is the # of unmasked points (this is an integer)*/
    nrows = 0;
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	nrows += (int) q1_mask[k][l][m];
      }
    }
    //printf("nrows: %d\n", nrows);
    /* start SVD matrix HERE */
    /* since the number of points change for each plane, we have to 
       re-allocate memory on each pass
    */
    /* remember A is column-major, and indexes [x,y] (col,row)!! */
    A = dmatrix_colmajor(nrows, 3);
    col = (double *) calloc(nrows, sizeof(double));
    rc = 0;
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	if(q1_mask[k][l][m]) {
	  A[0][rc] = (double) (m-n_fe/2);
	  A[1][rc] = (double) k;
	  A[2][rc] = 1.0;
	  col[rc++] = 0.5 * phsvol[k][l][m];
	}
      }
    }

    dsolve_svd(*A, col, soln, nrows, 3);
    /* do stuff with soln here */
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	zarg = (m-n_fe/2)*soln[0] + k*soln[1] + soln[2];
	/* The correction will be exp(-j*zarg) */
	pcor_vol[k][l][m][0] = cos(zarg);
	pcor_vol[k][l][m][1] = -sin(zarg);
      }
    }
    free_dmatrix(A);
    free(col);
  }
  
  apply_phase_correction(***image->data, **pcor_vol, n_fe, dsize, image->n_vol);

  free(soln);
  free_d3tensor(sigma);
  free_d3tensor(q1_mask);
  free_d3tensor(phsvol);
  free_c3tensor(pcor_vol);
  return;
}

/* Applies a complex volume corrector to the data, row by row. */
void apply_phase_correction(fftw_complex *data, fftw_complex *corrector,
			    int rowsize, int volsize, int nvols)
{
  int k, l, m, nrows = volsize/rowsize;
  fftw_complex *d, *c, *irow;
  double re1, re2, im1, im2;
  irow = (fftw_complex *) fftw_malloc(rowsize * sizeof(fftw_complex));
  for(k=0; k<nrows; k++) {
    c = corrector + k*rowsize;
    for(l=0; l<nvols; l++) {
      d = data + (l*volsize) + k*rowsize;
      fft1d(d, irow, rowsize, rowsize, INVERSE);
      for(m=0; m<rowsize; m++) {
	re1 = irow[m][0];
	im1 = irow[m][1];
	re2 = c[m][0];
	im2 = c[m][1];
	irow[m][0] = re1*re2 - im1*im2;
	irow[m][1] = re1*im2 + re2*im1;
      }
      /* transform the corrected data back into kspace (directly into d) */
      fft1d(irow, d, rowsize, rowsize, FORWARD);
    }
  }
  fftw_free(irow);
}


void unwrap_ref_volume(double *uphase, fftw_complex ***vol, 
		       int zdim, int ydim, int xdim, int xstart, int xstop)
{
  
  int k, l, m;
  int zerosl;
  double *scut, re, im, foo, height;
  double pi = acos(-1.0);
  float *wrplane, *uwplane;
  double ***phase;

  scut = (double *) malloc(zdim * sizeof(double));
  for(k=0; k<zdim; k++) {
    re = vol[k][ydim/2][xdim/2][0];
    im = vol[k][ydim/2][xdim/2][1];
    scut[k] = sqrt(re*re + im*im);
  }
  foo = array_max(scut, zdim, &zerosl);
  phase = d3tensor_alloc(zdim, ydim, xdim);
  angle(**phase, (const fftw_complex *) **vol, zdim*ydim*xdim);
  
  wrplane = (float *) malloc((zdim*xdim) * sizeof(float));
  uwplane = (float *) malloc((zdim*xdim) * sizeof(float));
  for(l=0; l<ydim; l++) {
    /* put mu-plane of wrapped phase vol into wrplane, then unwrap and 
       correct for any level offset, and finally put the data back 
       into uphase array */
    for(k=0; k<zdim; k++) {
      for(m=0; m<xdim; m++) {
	/* the mu-plane has dimension (zdim,xdim) */
	wrplane[k*xdim + m] = phase[k][l][m];
      }
    }
    doUnwrap(wrplane, uwplane, zdim, xdim);
    /* find height at the zerosl (found above) row, and where x = 0; */
    height = uwplane[zerosl*xdim + xdim/2];
    height = (double) ( (int) ((height + SIGN(height)*pi)/(2*pi)) );
    for(k=0; k<zdim; k++) {
      for(m=0; m<xdim; m++) {
	uphase[(k*ydim + l)*xdim + m] = uwplane[k*xdim + m] - 2*pi*height;
      }
    }
  }
  
}


void reverse_fe(fftw_complex *z, int n_fe, int len_z) {
  fftw_complex *dp;
  double re, im;
  int fe, k, m, nrows = len_z/n_fe;
  for(k=0; k<nrows; k++) {
    dp = z + k*n_fe;
    for(m=0; m<n_fe/2; m++) {
      re = dp[m][0];
      im = dp[m][1];
      dp[m][0] = dp[n_fe-m-1][0];
      dp[m][1] = dp[n_fe-m-1][1];
      dp[n_fe-m-1][0] = re;
      dp[n_fe-m-1][1] = im;
    }
  }
}

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
   
   matrix A is assumed to be in column major format (columns are 
   contiguous in memory)
*/

void dsolve_svd(double *A, double *y, double *x, int M, int N)
{
  char JOBZ = 'S';
  int LDA, LDU, LDVT, LWORK, IWORK, LRWORK, INFO, ns, k;
  double *is, *s, *u, *vt, *work, *vec1, *vec2, cond;
  enum CBLAS_ORDER Order;
  enum CBLAS_TRANSPOSE Trans1, Trans2;

  LDA = M;
  LDU = M;
  /* u should be shaped (m, ncol) ncol >= n */
  u = (double *) malloc(LDU * MIN(M,N) * sizeof(double));
  LDVT = MIN(M,N);
  /* vt should be shaped (ldvt, n) */
  vt = (double *) malloc(LDVT * N * sizeof(double));
  LWORK = -1;
  IWORK = 8*MIN(M,N);
  ns = MIN(M,N);
  work = (double *) malloc(1 * sizeof(double));
  s = (double *) malloc(sizeof(double)*ns);
  /* this run simply returns the best value for LWORK in work[0] */
  dgesdd_(&JOBZ, &M, &N, A, &LDA, s, u, &LDU, vt, &LDVT,
	  work, &LWORK, &IWORK, &INFO);
  LWORK = (int) work[0];
  free(work);
  work = (double *) malloc(LWORK * sizeof(double));
  dgesdd_(&JOBZ, &M, &N, A, &LDA, s, u, &LDU, vt, &LDVT,
	  work, &LWORK, &IWORK, &INFO);  
  if(INFO != 0) {
    printf("some error in SVD routine\n");
    exit(1);
  }

  /* So U is shaped (M, MIN(M,N))
     and Vt is shaped (MIN(M,N), N)
  */

  /* now x = dot(v, dot(inv(s), dot(ut, y)))
     we can send the CBLAS the order to transpose the matrices if needed */
  is = (double *) calloc((ns*ns), sizeof(double));
  for(k=0; k<ns; k++) is[k + k*ns] = 1.0/s[k];
  
  Order = CblasColMajor;
  Trans1 = CblasTrans;
  Trans2 = CblasNoTrans;

  vec1 = (double *) calloc(N, sizeof(double));
  vec2 = (double *) calloc(ns, sizeof(double));
  /*  keep doing the level-2 blas product and accumulated in vecs */
  /* this is vec1 <-- dot(ut, y) */
  cblas_dgemv(Order, Trans1, M, MIN(M,N), 1.0, u, M, y, 1, 0.0, vec1, 1);
  /* this is vec2 <-- dot(inv(s), vec1) */
  cblas_dgemv(Order, Trans2, ns, ns, 1.0, is, ns, vec1, 1, 0.0, vec2, 1);
  /* this is x <-- dot(v, vec2) */
  cblas_dgemv(Order, Trans1, MIN(M,N), N, 1.0, vt, MIN(M,N), vec2, 1, 0.0, x, 1);
  
  free(vec1);
  free(vec2);
  free(vt);
  free(u);
  free(s);
  free(work);
  free(is);
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

void maskbyfit(double *line, double *sigma, double *mask, double tol, 
	       double tol_growth, int len)
{
  
  int k, n;
  double m, b, res, mask_start, mask_end, fit;
  double *line_um, *sigma_um, *x_um;
  
  mask_start = cblas_dasum(len, mask, 1);
  if (!mask_start) {
    return;
  }
  line_um = (double *) malloc(mask_start * sizeof(double));
  sigma_um = (double *) malloc(mask_start * sizeof(double));
  x_um = (double *) malloc(mask_start * sizeof(double));
  n=0;
  for(k=0; k<len; k++) {
    if(mask[k]) {
      line_um[n] = line[k];
      sigma_um[n] = sigma[k];
      x_um[n++] = (double) k;
    }
  }
  linReg(line_um, x_um, sigma_um, (int) mask_start, &m, &b, &res);
  
  free(line_um);
  free(sigma_um);
  free(x_um);
  
  /* so wherever the diff between line and the fit are greater than
     (tol * res), mask that point */
  for(k=0; k<len; k++) {
    fit = ((double) k)*m + b;
    if ( ABS(line[k] - fit) > tol*res ) mask[k] = 0.0;
  }
  mask_end = cblas_dasum(len, mask, 1);
  /* limiting case */
  if (mask_end == mask_start) return;
  /* recursive case */
  else maskbyfit(line, sigma, mask, tol*tol_growth, tol_growth, len);
}


