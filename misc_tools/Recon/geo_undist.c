#include "recon.h"
#include "data.h"
#include "ops.h"

extern int zgesv_(int *n, int *nrhs, fftw_complex *a, 
		  int *lda, int *ipiv, fftw_complex *b, int *ldb, int *info);


/**************************************************************************
* geo_undistort                                                           *
*                                                                         *
*  An operation on k-space data                                           *
**************************************************************************/
 
void geo_undistort(image_struct *image, op_struct op)
{                        
  fftw_complex ****kern;
  double lambda;
  printf("Hello from geo_undistort \n");
  
  lambda = atof(op.param_1);
  printf("lambda is %2.3f ... \n", lambda);
  /* CHECK for existence of fieldmap before going anywhere */
  kern = c4tensor_alloc(image->n_slice, image->n_fe, 
			image->n_pe, image->n_pe);
  
  get_kernel(kern, image->fmap, image->mask, image->Tl, 
	     image->n_slice, image->n_fe, image->n_pe);


  /* do solutions here! */
  
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



void zsolve_regularized(fftw_complex *A, fftw_complex *y, fftw_complex *x,
			int M, int N, double lambda)
{
  
  fftw_complex *A2, *Atmp1;
  int k, INFO;
  static const double oneD[2] = {1.0, 0.0};
  static const double zeroD[2] = {0.0, 0.0};
  double lm_sq[2] = {0.0, 0.0};
  enum CBLAS_ORDER order;
  enum CBLAS_TRANSPOSE trans1, trans2;
  
  lm_sq[0] = lambda*lambda;
  
  trans1 = CblasConjTrans;
  trans2 = CblasNoTrans;
  order = CblasColMajor;

  Atmp1 = (fftw_complex *) fftw_malloc((M*N) * sizeof(fftw_complex));
  /* use A2 as the identity matrix for right now */
  A2 = (fftw_complex *) fftw_malloc((N*N) * sizeof(fftw_complex));
  //y2 = (fftw_complex *) fftw_malloc(N * sizeof(fftw_complex));

  bzero(A2, (N*N)*sizeof(fftw_complex));
  for(k=0; k<N*N; k++) A2[k][0] = 1.0;
  
  memmove(Atmp1, A, M*N*sizeof(fftw_complex));
  /* want to get A2 <-- (A')x(A) + (lm_sq)*I ... 
     also want to leave the result in
     column major so it can be used by LAPACK! Therefore, since A is already
     A' from a column-major perspective, do the operation in col-major, 
     don't transpose the left-hand matrix, and do transpose the right-hand 
     matrix. (DOES THIS WORK WITH HERMITIAN XPOSE?? I THINK YES)
  */
  cblas_zgemm(order, trans1, trans2, 
	      N, M, N, oneD, 
	      (void *) A, N, 
	      (void *) Atmp1, M,
	      lm_sq, (void *) A2, N);
  /* multiply data vector y by A', stick it in x*/
  cblas_zgemv(order, trans1, 
	      N, M, oneD,
	      (void *) A, N,
	      (void *) y, 1, zeroD, (void *) x, 1);
  /* solve the system! */
  zgesv_(&N, &N, A, &N, &N, x, &N, &INFO);
  
  fftw_free(Atmp1);
  fftw_free(A2);
}

  
