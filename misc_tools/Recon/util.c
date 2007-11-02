#include "recon.h"
#include "util.h"
#include "data.h"

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
/* void fft1d(fftw_complex *zin, fftw_complex *zout,  */
/* 	   int len_xform, int len_z, int direction) */
/* { */
/*   fftw_plan FT1D; */
/*   double tog = 1.0; */
/*   fftw_complex *dp_in, *dp_out; */
/*   int x, k, nxforms = len_z/len_xform; */

/*   FT1D = fftw_plan_dft_1d(len_xform, zin, zout, direction,  */
/* 			  FFTW_ESTIMATE | FFTW_PRESERVE_INPUT); */
  
/*   for(x=0; x<nxforms; x++) { */
/*     dp_in = zin + x*len_xform; */
/*     dp_out = zout + x*len_xform; */
/* /\*     FT1D = fftw_plan_dft_1d(len_xform, dp_in, dp_out, direction,  *\/ */
/* /\* 			     FFTW_ESTIMATE | FFTW_PRESERVE_INPUT); *\/ */
/*     for(k=0; k<len_xform; k++) { */
/*       dp_in[k][0] *= tog; */
/*       dp_in[k][1] *= tog; */
/*       tog *= -1.0; */
/*     } */
/*     fftw_execute_dft(FT1D, dp_in, dp_out); */
/* /\*     fftw_destroy_plan(FT1D);    *\/ */
/*     tog = 1.0; */
/*     for(k=0; k<len_xform; k++) { */
/*       /\* undo the modulation in both spaces *\/ */
/*       if(dp_in != dp_out) { */
/* 	dp_in[k][0] *= tog; */
/* 	dp_in[k][1] *= tog; */
/*       } */
/*       /\* FFTW does not normalize on the inverse, so do it here *\/ */
/*       if(direction == INVERSE) { */
/* 	dp_out[k][0] *= (tog/ (double) len_xform); */
/* 	dp_out[k][1] *= (tog/ (double) len_xform); */
/*       } else { */
/* 	dp_out[k][0] *= tog; */
/* 	dp_out[k][1] *= tog; */
/*       } */
/*       tog *= -1.0; */
/*     } */
/*   } */
/*   fftw_destroy_plan(FT1D); */
/*   fftw_cleanup(); */
/* } */

void fft1d(fftw_complex *zin, fftw_complex *zout, 
	   int len_xform, int len_z, int direction)
{
  fftw_plan FT1D;
  double tog = 1.0;
  int x, k, nxforms = len_z/len_xform;
  //n[0] = len_xform;
  // n[0] = N; howmany = nxforms; idist=N; istride=1; inembed = NULL
  FT1D = fftw_plan_many_dft(1, &len_xform, nxforms, zin, NULL, 1, len_xform,
			    zout, NULL, 1, len_xform, direction,
			    FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
  // modulate the input
  for(k=0; k<len_z; k++) {
    zin[k][0] *= tog;
    zin[k][1] *= tog;
    tog *= -1.0;
  }
  fftw_execute(FT1D);
  tog = 1.0;
  // demodulate output (and restore input if it's a separate array)
  for(k=0; k<len_z; k++) {
    if(zin != zout) {
      zin[k][0] *= tog;
      zin[k][1] *= tog;
    }
    if(direction == INVERSE) {
      zout[k][0] *= (tog/ (double) len_xform);
      zout[k][1] *= (tog/ (double) len_xform);
    } else {
      zout[k][0] *= tog;
      zout[k][1] *= tog;
    }
    tog *= -1.0;
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


/* Applies a complex volume corrector to the data, row by row. */
void apply_phase_correction(fftw_complex *data, fftw_complex *corrector,
			    int rowsize, int volsize, int nvols)
{
  int k, l, m, nrows = volsize/rowsize;
  fftw_complex *d;
  double re1, re2, im1, im2;
  fft1d(data, data, rowsize, volsize*nvols, INVERSE);
  for(m=0; m<volsize; m++) {
    re2 = corrector[m][0];
    im2 = corrector[m][1];
    for(l=0; l<nvols; l++) {
      d = data + l*volsize;
      re1 = d[m][0];
      im1 = d[m][1];
      d[m][0] = re1*re2 - im1*im2;
      d[m][1] = re1*im2 + re2*im1;
    }
  }
  fft1d(data, data, rowsize, volsize*nvols, FORWARD);
}

/* unwrap either the even-row segment or the odd-row segment of the */
/* phase of vol */
void unwrap_ref_volume(double *uphase, fftw_complex ***vol, 
		       int zdim, int ydim, int xdim, int seg)
{
  
  int k, l, m;
  int zerosl;
  double *s_line, re, im, foo, height;
  double pi = acos(-1.0);
  float *wrplane, *uwplane;
  double ***phase;

  s_line = (double *) malloc(zdim * sizeof(double));
  for(k=0; k<zdim; k++) {
    re = vol[k][ydim/2][xdim/2][0];
    im = vol[k][ydim/2][xdim/2][1];
    s_line[k] = sqrt(re*re + im*im);
  }
  foo = array_max(s_line, zdim, &zerosl);
  /* easiest just to get everything */
  phase = d3tensor_alloc(zdim, ydim, xdim);
  angle(**phase, (const fftw_complex *) **vol, zdim*ydim*xdim);
  
  wrplane = (float *) malloc((zdim*xdim) * sizeof(float));
  uwplane = (float *) malloc((zdim*xdim) * sizeof(float));
  for(l=0; l<ydim/2; l++) {
    /* put mu-plane of wrapped phase vol into wrplane, then unwrap and 
       correct for any level offset, and finally put the data back 
       into uphase array */
    for(k=0; k<zdim; k++) {
      for(m=0; m<xdim; m++) {
	/* the mu-plane has dimension (zdim,xdim) */
	wrplane[k*xdim + m] = phase[k][2*l+seg][m];
      }
    }
    phase_unwrap_2D(wrplane, uwplane, NULL, zdim, xdim);
    /* find height at the zerosl (found above) row, and where x = 0; 
       The idea is that this abs(phs[zerosl,zeropt]) < PI, so adjust the
       whole plane to fit that constraint.
    */
    height = uwplane[zerosl*xdim + xdim/2];
    height = (double) ( (int) ((height + SIGN(height)*pi)/(2*pi)) );
    for(k=0; k<zdim; k++) {
      for(m=0; m<xdim; m++) {
	uphase[(k*ydim/2 + l)*xdim + m] = uwplane[k*xdim + m] - 2*pi*height;
      }
    } 
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

/* This function fills in a (pre-allocated) mask by measuring the  */
/* quality of the phase using a 2nd derivative. In this case, the  */
/* mask should filled with 1s (at least on the rims) since the 2nd */
/* derivative is undefined there. */
void qual_map_mask_3d(double ***phs, double ***mask, 
		      int nz, int ny, int nx, double pct)
{
  
  double ***d2, *d2_sort, *tmp;
  double sum,  minbad;
  /* this is an emperical cut off, not the user defined one */
  double cutoff = 2.0;
  int i, j, k, pctile_idx;
  int (*comp) ();  

  d2 = d3tensor_alloc(nz-2, ny-2, nx-2);
  d2_sort = (double *) malloc((nz-2)*(ny-2)*(nx-2)*sizeof(double));
  
  /* fill in the outside shell of the mask, since we can't compute  */
  /* the 2nd derivative there */
  /* maybe it should just be filled in y=0 and y=ny-1, since the BPC */
  /* method is eventually only going to solve for each y value and   */
  /* only needs a valid mask for each z-x plane */
/*   for(k=0; k<nz; k+=(nz-1))  */
/*     for(j=0; j<ny; j++)  */
/*       for(i=0; i<nx; i++)  */
/* 	mask[k][j][i] = 1.0; */
  for(j=0; j<ny; j+=(ny-1))
    for(k=0; k<nz; k++)
      for(i=0; i<nx; i++)
	mask[k][j][i] = 1.0;
/*   for(i=0; i<nx; i+=(nx-1)) */
/*     for(k=0; k<nz; k++) */
/*       for(i=0; i<nx; i++) */
/* 	mask[k][j][i] = 1.0; */


  d2_xyz(phs, d2, nz, ny, nx, &minbad);
  
  /* The first cutoff throws out all points where the 2nd derivative */
  /* is bigger than 2 + the min badness */
  cutoff += minbad;
  printf("minbad: %f\n", minbad);
  printf("first cutoff: %f\n", cutoff);
  sum = 0.0;
  for(k=1; k<nz-1; k++) {
    for(j=1; j<ny-1; j++) {
      for(i=1; i<nx-1; i++) {
	if (d2[k-1][j-1][i-1] <= cutoff) {
	  mask[k][j][i] = 1.0;
	  sum += 1.0;
	} else {
	  mask[k][j][i] = 0.0;
	}
      }
    }
  }
  memmove(d2_sort, **d2, (nz-2)*(ny-2)*(nx-2)*sizeof(double));
  comp = &comparator;
  qsort(d2_sort, (nz-2)*(ny-2)*(nx-2), sizeof(double), comp);

  /* The second cutoff is where the badness is less than the Pth percentile */
  /* of remaining (unmasked) badnesses--so we can only consider the first */
  /* sum(mask) points in the sorted d2 */
  pctile_idx = (int) ( sum*pct/100. + 0.5 );
  cutoff = d2_sort[pctile_idx];
  printf("2nd cutoff: %f (%dth out of %d pts considered--%2.1fth percntile)\n", cutoff, pctile_idx, (int) sum, pct);
  for(k=1; k<nz-1; k++) {
    for(j=1; j<ny-1; j++) {
      for(i=1; i<nx-1; i++) {
	mask[k][j][i] = d2[k-1][j-1][i-1] > cutoff ? 0.0 : 1.0;
      }
    }
  }
  
  free_d3tensor(d2);
  free(d2_sort);
}

/* This function is like the previous, except that the quality measure    */
/* will be averaged over the y-direction (corresponding to unbal-PC data) */
void qual_map_mask_2d(double ***phs, double **mask,
		      int nz, int ny, int nx, double pct)
{
  double ***d2, **d2_mean, *d2_sort;
  double sum;
  double cutoff = 0.2;
  int i, j, k, pctile_idx, d2_ysize;
  int (*comp) ();
  d2_ysize = MAX(ny-2,1);
  d2 = d3tensor_alloc(nz-2, d2_ysize, nx-2);
  d2_mean = dmatrix(nz-2, nx-2);
  d2_sort = (double *) malloc((nz-2)*(nx-2)*sizeof(double));

  d2_xyz(phs, d2, nz, ny, nx, NULL);
  /* accumulate over the Y-dimension */
  for(j=0; j<d2_ysize; j++) {
    for(k=0; k<nz-2; k++) {
      for(i=0; i<nx-2; i++) {
	d2_mean[k][i] += d2[k][j][i] / (double)d2_ysize;
      }
    }
  }
  memmove(d2_sort, *d2_mean, (nz-2)*(nx-2)*sizeof(double));
  comp = &comparator;
  qsort(d2_sort, (nz-2)*(nx-2), sizeof(double), comp);
  /* The first cutoff throws out all points where the 2nd derivative */
  /* is bigger than 0.2 + the min badness */
  cutoff += d2_sort[0];
  sum = 0.0;
  for(k=1; k<nz-1; k++) {
    for(i=1; i<nx-1; i++) {
      if (d2_mean[k-1][i-1] <= cutoff) {
	mask[k][i] = 1.0;
	sum += 1.0;
      } else {
	mask[k][i] = 0.0;
      }
    }
  }
  /* The second cutoff is where the badness is less than the Pth percentile */
  /* of remaining (unmasked) badnesses--so we can only consider the first */
  /* sum(mask) points in the sorted d2 */
  pctile_idx = (int) (sum*pct/100. + 0.5);
  cutoff = d2_sort[pctile_idx];
  for(k=1; k<nz-1; k++) {
    for(i=1; i<nx-1; i++) {
      mask[k][i] = d2_mean[k-1][i-1] > cutoff ? 0.0 : 1.0;
    }
  }
  free_d3tensor(d2);
  free_dmatrix(d2_mean);
  free(d2_sort);
}


void d2_xyz(double ***box, double ***d2, int nz, int ny, int nx, double *min)
{
  double dbox, sum, minbad;
  int i, j, k;
  /* 2nd derivative formula is: */
  /* d2[n] = (d[n+1]-d[n]) - (d[n]-d[n-1]) = d[n+1] - 2*d[n] + d[n-1] */
  minbad = 1e6;
  
  /* if ny < 3, can't do a 2nd derivative in the Y-direction */
  if(ny < 3) {
    for(k=1; k<nz-1; k++) {
      for(i=1; i<nx-1; i++) {
	sum = 0.0;
	/* x-diff */
	dbox = box[k][0][i+1] + box[k][0][i-1] - 2*box[k][0][i];
	sum += dbox*dbox;
	/* z-diff */
	dbox = box[k+1][0][i] + box[k-1][0][i] - 2*box[k][0][i];
	sum += dbox*dbox;
	if (sum < minbad) minbad = sum;
	d2[k-1][0][i-1] = sqrt(sum);
      }
    }
  } else {
    for(k=1; k<nz-1; k++) {
      for(j=1; j<ny-1; j++) {
	for(i=1; i<nx-1; i++) {
	  sum = 0.0;
	  /* x-diff */
	  dbox = box[k][j][i+1] + box[k][j][i-1] - 2*box[k][j][i];
	  sum += dbox*dbox;
	  /* y-diff */
	  dbox = box[k][j+1][i] + box[k][j-1][i] - 2*box[k][j][i];
	  sum += dbox*dbox;
	  /* z-diff */
	  dbox = box[k+1][j][i] + box[k-1][j][i] - 2*box[k][j][i];
	  sum += dbox*dbox;
	  if (sum < minbad) minbad = sum;
	  d2[k-1][j-1][i-1] = sqrt(sum);
	}
      }
    }
  }

  if (min != NULL) *min = sqrt(minbad);
}
