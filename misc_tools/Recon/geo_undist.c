#include "recon.h"
#include "data.h"
#include "ops.h"

/* useful constants for cblas_zgemm */
static const double oneD[2] = {1.0, 0.0};
static const double zeroD[2] = {0.0, 0.0};

/* the LAPACK linear solver */
extern int zgesv_(int *n, int *nrhs, fftw_complex *a, 
		  int *lda, int *ipiv, fftw_complex *b, int *ldb, int *info);

/**************************************************************************
* geo_undistort                                                           *
*                                                                         *
*  An operation on k-space data                                           *
**************************************************************************/

void geo_undistort(image_struct *image, op_struct op)
{                        
  fftw_complex ****kern, *id, *dchunk, *dist_chunk, *soln_chunk;
  double lambda;
  int k, l, m, n, n_vol, n_slice, n_pe, n_fe;
  FILE *fp;

  n_vol = image->n_vol;
  n_slice = image->n_slice;
  n_pe = image->n_pe;
  n_fe = image->n_fe;
  
  lambda = atof(op.param_1);
  printf("lambda is %2.3f ... \n", lambda);
  
  /* CHECK for existence of fieldmap before going anywhere */
  if (!image->fmap) {
    printf("fieldmap never was computed! doing nothing...\n");
    return;
  }
  
  /* MEMORY allocation */
  kern = c4tensor_alloc(n_slice, n_fe, n_pe, n_pe);
  //dchunk = c3tensor_alloc(n_vol, n_pe, n_fe);
  dchunk = (fftw_complex *) fftw_malloc(n_vol*n_pe*n_fe * sizeof(fftw_complex));
  dist_chunk = (fftw_complex *) fftw_malloc(n_pe*n_vol * sizeof(fftw_complex));
  soln_chunk = (fftw_complex *) fftw_malloc(n_pe*n_vol * sizeof(fftw_complex));
  id = (fftw_complex *) fftw_malloc((n_pe*n_pe) * sizeof(fftw_complex));
  /* FREE THESE */

  bzero(id, (n_pe*n_pe*sizeof(fftw_complex)));
  for(k=0; k<n_pe; k++) id[k + k*n_pe][0] = 1.0;

  get_kernel(kern, image->fmap, image->mask, image->Tl, n_slice, n_fe, n_pe);
/*   fp = fopen("kern", "wb"); */
/*   fwrite(***kern, n_slice*n_fe*n_pe*n_pe, sizeof(fftw_complex), fp); */
/*   fclose(fp); */
/*   fp = fopen("fmap", "wb"); */
/*   fwrite(**image->fmap, n_slice*n_fe*n_pe, sizeof(double), fp); */
/*   fclose(fp); */
/*   fp = fopen("mask", "wb"); */
/*   fwrite(**image->mask, n_slice*n_fe*n_pe, sizeof(double), fp); */
/*   fclose(fp); */
  printf("computing inverse operators...");
  for(k=0; k<n_slice; k++) {
    for(l=0; l<n_fe; l++) {
      zsolve_regularized(*(kern[k][l]), id, *(kern[k][l]), n_pe, n_pe,
			 n_pe, lambda);
    }
  }
  printf(" done\n");
  /* do solutions here! */

  /* work one plane (slice) at a time:
     1) (inverse) xform data along n1->q1 (ksp->isp)
     2) for each column of data[:][q1] say col <-- dot(iK[sl][q1], data[:][q1])
     3) (forward) xform data back to ksp
     4) 
  */
  for(l=0; l<n_slice; l++) {
    /* slice across array at slice=l and put the data in dchunk */
    for(k=0; k<n_vol; k++) {
      memmove(dchunk + (k*n_pe*n_fe),
	      *(image->data[k][l]),
	      n_pe*n_fe*sizeof(fftw_complex));
    }
    /* inverse transform along the FE dimension */
    fft1d(dchunk, dchunk, n_fe, n_vol*n_pe*n_fe, INVERSE);
    /* for every FE point, apply the inverse operator.. 
       do this by making a matrix that is shaped (n_pe x n_vol) and
       left-multiplying it by the (n_pe x n_pe) operator */
    for(n=0; n<n_fe; n++) {
      
      for(m=0; m<n_pe; m++) {
	for(k=0; k<n_vol; k++) {
	  /* this is equivalently soln_chunk[m][k] = dchunk[k][m][n] */
	  memmove(dist_chunk + (m*n_vol + k),
		  dchunk + ((k*n_pe + m)*n_fe + n),
		  sizeof(fftw_complex));
	}
      }
      /* apply op (NPExNPE)x(NPExNVOL) --> (M,N,K) are (NPE,NPE,NPE) */
      if (n_vol > 1) {
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    n_pe, n_pe, n_pe, oneD,
		    (void *) *(kern[l][n]), n_pe,
		    (void *) dist_chunk, n_pe,
		    zeroD, (void *) soln_chunk, n_pe);
      } else {
	cblas_zgemv(CblasRowMajor, CblasNoTrans,
		    n_pe, n_pe, oneD,
		    (void *) *(kern[l][n]), n_pe,
		    (void *) dist_chunk, 1, 
		    zeroD, (void *) soln_chunk, 1);
      }
      /* put solution back into dchunk */
      for(m=0; m<n_pe; m++) {
	for(k=0; k<n_vol; k++) {
	  memmove(dchunk + ((k*n_pe + m)*n_fe + n),
		  soln_chunk + (m*n_vol + k),
		  sizeof(fftw_complex));
	}
      } 
    }
    printf("solved slice %d\n", l);
    fft1d(dchunk, dchunk, n_fe, n_vol*n_pe*n_fe, FORWARD);
    for(k=0; k<n_vol; k++) {
      memmove(*(image->data[k][l]),
	      dchunk + (k*n_pe*n_fe),
	      n_pe*n_fe*sizeof(fftw_complex));
    }
  }

  fftw_free(soln_chunk);
  fftw_free(dist_chunk);
  fftw_free(dchunk);
  fftw_free(id);
  free_c4tensor(kern);

  return;
}        

void get_kernel(fftw_complex ****kernel, double ***fmap, double ***vmask,
		double Tl, int ns, int nr, int nc)
{
  
  int N2,N2P,M1,M2,n2,n2p,q1,q2,sl,idx;
  double re, im, zarg, tn2;
  double pi = acos(-1.0);
  double *dp, cn;
  fftw_complex *svals;
  fftw_complex ***basis_xform;
  //fftw_complex ***kern;
  fftw_complex ***e2;
  fftw_complex *sum;

  sum = (fftw_complex *) malloc(sizeof(fftw_complex));
  N2 = nr;
  N2P = nr;
  M1 = nc;
  M2 = nr;

  basis_xform = c3tensor_alloc(N2,N2P,M2);
  for(n2 = 0; n2 < N2; n2++) {
    
    for(n2p = 0; n2p < N2P; n2p++) {
      
      for(q2 = 0; q2 < M2; q2++) {
	zarg = (2.0 * pi * (n2p - n2) * (q2 - M2/2))/(double) M2;
	basis_xform[n2][n2p][q2][0] = cos(zarg)/M2;
	basis_xform[n2][n2p][q2][1] = sin(zarg)/M2;
      }

    }

  }

  for(sl=0; sl < ns; sl++) {
    printf("starting kernel calc for sl=%d ... \n", sl);
    e2 = c3tensor_alloc(N2,M2,M1);
    for(n2 = 0; n2 < N2; n2++) {
      tn2 = (n2 - N2/2) * Tl;
      for(q2 = 0; q2 < M2; q2++) {
      
	for(q1 = 0; q1 < M1; q1++) {
	  // nr rows per slice
	  // nc pts per row
	  if(vmask[sl][q2][q1]) {
	    zarg = tn2 * fmap[sl][q2][q1];
	    e2[n2][q2][q1][0] = cos(zarg);
	    e2[n2][q2][q1][1] = sin(zarg);
	  } else {
	    e2[n2][q2][q1][0] = 0.0;
	    e2[n2][q2][q1][1] = 0.0;
	  }
	}
      }
    }
  
    //kern = f3tensor(M1,N2,N2P);
    // k is (M1,N2,N2P)
    // basis_xform is (N2,N2P,M2)
    // e2 is (N2,M2,M1)
    for(q1 = 0; q1 < M1; q1++) {
      for(n2 = 0; n2 < N2; n2++) {
	for(n2p = 0; n2p < N2P; n2p++) {
	  sum[0][0] = 0.0;
	  sum[0][1] = 0.0;
	  for(q2 = 0; q2 < M2; q2++) {
	    if(vmask[sl][q2][q1]) {
	      sum[0][0] += (basis_xform[n2][n2p][q2][0] * e2[n2][q2][q1][0])
		          -(basis_xform[n2][n2p][q2][1] * e2[n2][q2][q1][1]);

	      sum[0][1] += (basis_xform[n2][n2p][q2][1] * e2[n2][q2][q1][0])
		          +(basis_xform[n2][n2p][q2][0] * e2[n2][q2][q1][1]);
	    }
	  }
	  kernel[sl][q1][n2][n2p][0] = sum[0][0];
	  kernel[sl][q1][n2][n2p][1] = sum[0][1];
	}
      }
    }
  }
  free(basis_xform);
  free(e2);
  free(sum);
}

/* data shapes:
   A is MxN (usually square) (row-major)
   y is MxNRHS (row-major)
   x is NxNRHS (row-major)
*/

void zsolve_regularized(fftw_complex *A, fftw_complex *y, fftw_complex *x,
			int M, int N, int NRHS, double lambda)
{
  
  fftw_complex *A2, *A2_cm, *x_cm;
  int k, l, INFO, *IPIV;
  double lm_sq[2] = {0.0, 0.0};

  lm_sq[0] = lambda*lambda;
  
  //Atmp1 = (fftw_complex *) fftw_malloc((M*N) * sizeof(fftw_complex));
  /* use A2 as the identity matrix for right now */
  A2 = (fftw_complex *) fftw_malloc((N*N) * sizeof(fftw_complex));
  A2_cm = (fftw_complex *) fftw_malloc((N*N) * sizeof(fftw_complex));
  x_cm = (fftw_complex *) fftw_malloc((NRHS*N) * sizeof(fftw_complex));
  IPIV = (int *) malloc(N * sizeof(int));
  /* guess we need temp space for xpose */
  

  //y2 = (fftw_complex *) fftw_malloc(N * sizeof(fftw_complex));

  bzero(A2, (N*N)*sizeof(fftw_complex));
  for(k=0; k<N; k++) A2[k + k*N][0] = 1.0;
  
  /* want to get A2 <-- (A*)x(A) + (lm_sq)*I ... 
  */

  /* this only gives back the upper triangular part!! */
/*   cblas_zherk(CblasRowMajor, CblasUpper, CblasConjTrans, */
/* 	      N, N, 1.0, */
/* 	      (void *) A, M, */
/* 	      0.0, (void *) A2, N); */
	      
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
	      N, M, N, oneD,
	      (void *) A, N,
	      (void *) A, M,
	      lm_sq, (void *) A2, N);
  /* multiply data vector/matrix y by A*, stick it in x*/
  /* DOES THIS WORK WITH NRHS = 1? */
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
	      N, M, N, oneD,
	      (void *) A, M,
	      (void *) y, M, 
	      zeroD, (void *) x, N);
  /* solve the system! need to xpose ante and post to get into col-major:
     A2 is NxN, and x is NxNRHS
  */
  /* can this be done faster? */
  /* could we use the solution for hermitian positive definite matrix?? */
  for(k=0; k<N; k++) {
    for(l=0; l<N; l++) {
      /* rm idx = k*N + l; cm idx = l*N + k */
      memmove(A2_cm+(l*N+k), A2+(k*N+l), sizeof(fftw_complex));
    }
  }
  for(k=0; k<N; k++) {
    for(l=0; l<NRHS; l++) {
      memmove(x_cm+(l*N+k), x+(k*NRHS+l), sizeof(fftw_complex));
    }
  }
    
  zgesv_(&N, &NRHS, A2_cm, &N, IPIV, x_cm, &N, &INFO);

  /* only xpose back x (A2 was temporary) */
  for(k=0; k<N; k++) {
    for(l=0; l<NRHS; l++) {
      memmove(x+(k*NRHS+l), x_cm+(l*N+k), sizeof(fftw_complex));
    }
  }


  free(IPIV);
  fftw_free(x_cm);
  fftw_free(A2_cm);
  fftw_free(A2);
}

  
