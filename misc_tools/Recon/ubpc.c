#include "recon.h"
#include "util.h"
#include "data.h"

void unbal_phs_corr(image_struct *image, op_struct op)
{
  fftw_complex ***conjref, ***pcor_vol;
  double ***phsvol_ev, ***phsvol_od, **q1_mask_ev, **q1_mask_od;
  double **phsvol_mean_ev, **phsvol_mean_od;
  double **working_phs, **working_msk;
  double **A, *col, *soln;
  double re1, re2, im1, im2, zarg, sum1, sum2, b;
  int k, l, m, n_fe, n_pe, n_ref_rows, n_conj_rows, n_slice, dsize, nSVDrows, rc;

  n_fe = image->n_fe;
  n_pe = image->n_pe;
  /* With the Siemens data, the # of ref-scan rows may be variable */
  /* this should arguably be n_ref_rows = image->n_ref_rows */
  /* for now, we'll let it be an optional parameter, or n_pe by default */
  n_ref_rows = atof(op.param_1) ? atof(op.param_1) : n_pe;
  /* want n_conj_rows to be even */
  n_conj_rows = n_ref_rows%2 ? n_ref_rows-1 : n_ref_rows-2;
  n_slice = image->n_slice;
  dsize = n_slice * n_pe * n_fe;
  
  conjref = c3tensor_alloc(n_slice, n_conj_rows, n_fe);
  phsvol_ev = d3tensor_alloc(n_slice, n_conj_rows/2, n_fe);
  phsvol_od = d3tensor_alloc(n_slice, n_conj_rows/2, n_fe);
  phsvol_mean_ev = dmatrix(n_slice, n_fe);
  phsvol_mean_od = dmatrix(n_slice, n_fe);

  q1_mask_ev = dmatrix(n_slice, n_fe);
  q1_mask_od = dmatrix(n_slice, n_fe);

  pcor_vol = c3tensor_alloc(n_slice, n_pe, n_fe);
  soln = (double *) calloc(6, sizeof(double));

  /* do IFFT in-place */
  fft1d(**image->ref1, **image->ref1, n_fe, dsize, INVERSE);
  
  /* conjref[:,l,:] = ifft(ref)[:,l,:] * conj(ifft(ref)[:,l+1,:]) */
  for(l=0; l<n_conj_rows; l++) {
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	re1 = image->ref1[k][l][m][0];
	im1 = image->ref1[k][l][m][1];
	re2 = image->ref1[k][l+1][m][0];
	im2 = image->ref1[k][l+1][m][1];
	/* (re1+j*im1)*(re2-j*im2) = re1*re2+im1*im2 + j*(re2*im1-re1*im2) */
	conjref[k][l][m][0] = re1*re2 + im1*im2;
	conjref[k][l][m][1] = re2*im1 - re1*im2;
      }
    }
  }
  unwrap_ref_volume(**phsvol_ev, conjref, n_slice, n_conj_rows, n_fe, 0);
  unwrap_ref_volume(**phsvol_od, conjref, n_slice, n_conj_rows, n_fe, 1);
  
  free_c3tensor(conjref);
  
  /* get the 2d masks */
  qual_map_mask_2d(phsvol_ev, q1_mask_ev, n_slice, n_conj_rows/2, n_fe, 75.0);
  qual_map_mask_2d(phsvol_od, q1_mask_od, n_slice, n_conj_rows/2, n_fe, 75.0);
  /* look at the rows in the masks, if there are less than 4 pts, mask row */
  /* we'll also count the total number of unmasked pts, which is nSVDrows of  */
  /* the SVD matrix */
  nSVDrows = 0;
  for(k=0; k<n_slice; k++) {
    sum1 = 0.0; sum2 = 0.0;
    for(m=0; m<n_fe; m++) {
      sum1 += q1_mask_ev[k][m];
      sum2 += q1_mask_od[k][m];
    }
    nSVDrows += ((int) sum1 + (int) sum2);
    if(sum1 < 4) {
      for(m=0; m<n_fe; m++) q1_mask_ev[k][m] = 0.0;
      nSVDrows -= (int) sum1;
    }
    if(sum2 < 4) {
      for(m=0; m<n_fe; m++) q1_mask_od[k][m] = 0.0;
      nSVDrows -= (int) sum2;
    }
  }

  /* find the mean over the Y-axis */
  for(l=0; l<n_conj_rows/2; l++) {
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	phsvol_mean_ev[k][m] += 2 * phsvol_ev[k][l][m] / (double) n_conj_rows;
	phsvol_mean_od[k][m] += 2 * phsvol_od[k][l][m] / (double) n_conj_rows;
      }
    }
  }
  
  A = dmatrix(nSVDrows, 6);
  col = (double *) calloc(nSVDrows, sizeof(double));
  rc = 0;
  /* construct the SVD matrix with all evn eqs, then all odd eqs:  */
  /* phs(k,l,m) =  2*[m*A1 + k*A3 + A5] - [m*A2 + k*A4 + A6] (l-even) */
  /* phs(k,l,m) = -2*[m*A1 + k*A3 + A5] - [m*A2 + k*A4 + A6] (l-odd)  */
  for(l=0; l<2; l++) {
    working_phs = l ? phsvol_mean_od : phsvol_mean_ev;
    working_msk = l ? q1_mask_od : q1_mask_ev;
    b = l ? -2.0 : 2.0;
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	if(working_msk[k][m]) {
	  A[rc][0] = b*((double) (m-n_fe/2));
	  A[rc][2] = b*((double) k);
	  A[rc][4] = b;
	  A[rc][1] = (double) -(m-n_fe/2);
	  A[rc][3] = (double) -k;
	  A[rc][5] = (double) -1.0;
	  col[rc++] = working_phs[k][m];
	}
      }
    }
  }
  dsolve_svd(*A, col, soln, nSVDrows, 6);
  for(l=0; l<n_pe; l++) {
    b = l%2 ? -1.0 : 1.0;
    for(k=0; k<n_slice; k++) {
      for(m=0; m<n_fe; m++) {
	zarg = b*((m-n_fe/2)*soln[0] + k*soln[2] + soln[4]);
	pcor_vol[k][l][m][0] = cos(zarg);
	pcor_vol[k][l][m][1] = -sin(zarg);
      }
    }
  }

  apply_phase_correction(***image->data, **pcor_vol, n_fe, dsize, image->n_vol);

  free(soln);
  free_dmatrix(A);
  free(col);
  free_d3tensor(phsvol_ev);
  free_d3tensor(phsvol_od);
  free_dmatrix(phsvol_mean_ev);
  free_dmatrix(phsvol_mean_od);
  free_dmatrix(q1_mask_ev);
  free_dmatrix(q1_mask_od);
  free_c3tensor(pcor_vol);
  return;
}
