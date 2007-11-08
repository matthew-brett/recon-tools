#include "recon.h"
#include "util.h"
#include "data.h"

void bal_phs_corr(image_struct *image, op_struct op)
{
  fftw_complex ***conjref, ***invref1, ***invref2, *ir1, *ir2, *ir, ***pcor_vol;
  double ***phsvol_ev, ***phsvol_od, ***q1_mask_ev, ***q1_mask_od;
  double ***working_msk, ***working_phs;
  double **A, *col, *soln;
  double re1, re2, im1, im2, zarg;
  int k, l, m, n_fe, n_pe, n_slice, dsize, nrows, rc;

  n_fe = image->n_fe;
  n_pe = image->n_pe;
  n_slice = image->n_slice;
  dsize = n_slice * n_pe * n_fe;
  /* memory allocation */
  invref1 = c3tensor_alloc(n_slice, n_pe, n_fe);
  invref2 = c3tensor_alloc(n_slice, n_pe, n_fe);
  conjref = c3tensor_alloc(n_slice, n_pe, n_fe);  
  phsvol_ev = d3tensor_alloc(n_slice, n_pe/2, n_fe);
  phsvol_od = d3tensor_alloc(n_slice, n_pe/2, n_fe);

  q1_mask_ev = d3tensor_alloc(n_slice, n_pe/2, n_fe);
  q1_mask_od = d3tensor_alloc(n_slice, n_pe/2, n_fe);
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
  ir = **conjref;
  /* conjref[:,l,:] = ifft(ref1)[:,l,:] * conj(ifft(reverse_fe(ref2)))[:,l,:] */
  for(k=0; k<dsize; k++) {
    re1 = ir1[k][0];
    re2 = ir2[k][0];
    im1 = ir1[k][1];
    im2 = ir2[k][1];
    /* (re1+j*im1)*(re2-j*im2) = re1*re2+im1*im2 + j*(re2*im1-re1*im2) */
    ir[k][0] = re1*re2 + im1*im2;
    ir[k][1] = -re1*im2 + re2*im1;
  }
  
  /* unwrap the even and odd parts (forward and backward trajectories) */
  /* into separate blocks */
  unwrap_ref_volume(**phsvol_ev, conjref, n_slice, n_pe, n_fe, 0);
  unwrap_ref_volume(**phsvol_od, conjref, n_slice, n_pe, n_fe, 1);

  /* can get rid of invref stuff now */
  free_c3tensor(invref1);
  free_c3tensor(invref2);
  free_c3tensor(conjref);  

  /* get a mask based on the "quality" measure of the phase */
  qual_map_mask_3d(phsvol_ev, q1_mask_ev, n_slice, n_pe/2, n_fe, 90.0);
  qual_map_mask_3d(phsvol_od, q1_mask_od, n_slice, n_pe/2, n_fe, 90.0);

  /* For each mu in n_pe, solve for the planar fit of the surface. */
  for(l=0; l<n_pe; l++) {
    working_msk = l%2 ? q1_mask_od : q1_mask_ev;
    working_phs = l%2 ? phsvol_od  : phsvol_ev;
    /* The # of rows in A is the # of unmasked points (this is an integer)*/
    nrows = 0;
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	nrows += (int) working_msk[k][l/2][m];
      }
    }
    /* start SVD matrix HERE */
    /* since the number of points change for each plane, we have to 
       re-allocate memory on each pass
    */
    if (nrows < 10) {
      printf("uhoh");
    }
    A = dmatrix(nrows, 3);
    col = (double *) calloc(nrows, sizeof(double));
    rc = 0;
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	if(working_msk[k][l/2][m]) {
	  A[rc][0] = (double) (m-n_fe/2);
	  A[rc][1] = (double) k;
	  A[rc][2] = 1.0;
	  col[rc++] = 0.5 * working_phs[k][l/2][m];
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
  free_d3tensor(q1_mask_ev);
  free_d3tensor(q1_mask_od);
  free_d3tensor(phsvol_ev);
  free_d3tensor(phsvol_od);
  free_c3tensor(pcor_vol);
  return;
}

void reverse_fe(fftw_complex *z, int n_fe, int len_z) {
  fftw_complex *dp;
  double re, im;
  int k, m, nrows = len_z/n_fe;
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

