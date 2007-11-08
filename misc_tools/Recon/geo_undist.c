#include "recon.h"
#include "data.h"
#include "util.h"

/* useful constants for cblas_zgemm */
static const double oneD[2] = {1.0, 0.0};
static const double zeroD[2] = {0.0, 0.0};

/* the LAPACK linear solver */
extern int zgesv_(int *n, int *nrhs, fftw_complex *a, 
		  int *lda, int *ipiv, fftw_complex *b, int *ldb, int *info);

extern int zposv_(char *UPLO, int *N, int *NRHS, fftw_complex *a, int *LDA,
		  fftw_complex *b, int *LDB, int *INFO);

/**************************************************************************
* geo_undistort                                                           *
*                                                                         *
*  Corrects susceptibility errors in EPI data                             *
**************************************************************************/

void geo_undistort(image_struct *image, op_struct op)
{                        
  fftw_complex ****kern, *dchunk, *dist_chunk, *soln_chunk;
  double lambda;
  int k, l, m, n, n_vol, n_slice, n_pe, n_fe;

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
  /* ^^^ FREE THESE ^^^*/
  printf("getting kernel... ");
  get_kernel(kern, image->fmap, image->mask, image->Tl, n_slice, n_fe, n_pe);
  printf("done\ncomputing inverse operators... ");
  for(k=0; k<n_slice; k++) {
    for(l=0; l<n_fe; l++) {
      zregularized_inverse(*(kern[k][l]), n_pe, n_pe, lambda);
    }
  }
  printf("done\n");
  /* do solutions here! */

  /* work one plane (slice) at a time:
     1) (inverse) xform data along n1->q1 (ksp->isp)
     2) for each column of data[:][q1] say col <-- dot(iK[sl][q1], data[:][q1])
     3) (forward) xform data back to ksp
     4) 
  */
  printf("correcting data... ");
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
	  /* this is equivalently dist_chunk[m][k] = dchunk[k][m][n] */
	  memmove(dist_chunk + (m*n_vol + k),
		  dchunk + ((k*n_pe + m)*n_fe + n),
		  sizeof(fftw_complex));
	}
      }
      /* apply inv. op (NPExNPE)x(NPExNVOL) --> (M,N,K) are (NPE,N_VOL,NPE) */
      /* LDA,LDB,LDC = major stride size!!!
	 IE: LDA = NPE, LDB = NVOL, LDC = NVOL (since they are row-major)
      */
      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		  n_pe, n_vol, n_pe, oneD,
		  (void *) *(kern[l][n]), n_pe,
		  (void *) dist_chunk, n_vol,
		  zeroD, (void *) soln_chunk, n_vol);
      /* put solution back into dchunk */
      for(m=0; m<n_pe; m++) {
	for(k=0; k<n_vol; k++) {
	  memmove(dchunk + ((k*n_pe + m)*n_fe + n),
		  soln_chunk + (m*n_vol + k),
		  sizeof(fftw_complex));
	}
      } 
    }
    fft1d(dchunk, dchunk, n_fe, n_vol*n_pe*n_fe, FORWARD);
    for(k=0; k<n_vol; k++) {
      memmove(*(image->data[k][l]),
	      dchunk + (k*n_pe*n_fe),
	      n_pe*n_fe*sizeof(fftw_complex));
    }
  }
  printf("done. \n\n");
  fftw_free(soln_chunk);
  fftw_free(dist_chunk);
  fftw_free(dchunk);
  free_c4tensor(kern);

  return;
}        

/* Do this faster, here's the Python model:

t_n2 = (N.arange(64)-32)*Tl
pshifts = N.exp(-1.j*2*N.pi*N.outer((N.arange(64)-32), (N.arange(64)-32))/64)
F = bmask_xp[:,:,None,:] * N.exp(1.j*fmap_xp[:,:,None,:]*t_n2[None,None,:,None]) * pshifts

F is shaped (nslice, N1, N2, M2)

K = util.ifft(F)

K is shaped (nslice, N1, N2, N2P)
*/

void get_kernel(fftw_complex ****kernel, double ***fmap, double ***vmask,
		double Tl, int ns, int nr, int nc)
{

  double *t_n2;
  fftw_complex ****F;
  int N2, N2P, M1, M2, n2, m1, m2, sl;
  double zarg;
  double pi = acos(-1.0);

  // this isn't strictly true.. n2/n2p and m2 can be different
  // n2 refers to the image resolution and m2 refers to the fmap resolution
  N2 = N2P = nr;
  M2 = nr; // THIS SHOULD BE THE # of ROWS IN FMAP, but algorithm not ready
  M1 = nc; // THIS SHOULD BE THE # of COLS IN FMAP (should == image->n_fe)
  // maybe M2,m2 should stay Q2,q2 (so not to confuse with image dimension M2)

  t_n2 = (double *) malloc(N2 * sizeof(double));
  F = c4tensor_alloc(ns, M1, N2, M2);


  for(n2=0; n2<N2; n2++) t_n2[n2] = (double) (n2 - N2/2) * Tl;

  //fmap and vmask are coming in shaped (nslice, npe, nfe).. this calculation
  //will require indexing them as if they're transposed
  for(sl=0; sl<ns; sl++) {
    for(m1=0; m1<M1; m1++) {
      for(n2=0; n2<N2; n2++) {
	for(m2=0; m2<M2; m2++) {
	  if(vmask[sl][m2][m1]) {
	    // zarg is the phase(t) function minus a phase shift..
	    // this phase shift effects a time shift after the FFT
	    zarg = fmap[sl][m2][m1]*t_n2[n2] -				\
	      (2.0 * pi * ((double) (n2-N2/2) * (m2-M2/2) / (double) M2));
	    
	    F[sl][m1][n2][m2][0] = cos(zarg);
	    F[sl][m1][n2][m2][1] = sin(zarg);
	  } //else {
	    //kernel is already zero'd
	}
      }
    }
  }
  /* If the fmap is of a different resolution, this will need to be a
     M2 -> N2 transform (M2 >= N2).. so potentially, only a subset of
     computed frequencies will be kept. */
  fft1d(***F, ***kernel, M2, ns*M1*N2*M2, INVERSE);
  
  free(t_n2);
  free_c4tensor(F);
}
	  

/* void get_kernel(fftw_complex ****kernel, double ***fmap, double ***vmask, */
/* 		double Tl, int ns, int nr, int nc) */
/* { */
  
/*   int N2,N2P,M1,M2,n2,n2p,q1,q2,sl,idx; */
/*   double re, im, zarg, tn2; */
/*   double pi = acos(-1.0); */
/*   double *dp, cn; */
/*   fftw_complex *svals; */
/*   fftw_complex ***basis_xform; */
/*   //fftw_complex ***kern; */
/*   fftw_complex ***e2; */
/*   fftw_complex *sum; */

/*   sum = (fftw_complex *) malloc(sizeof(fftw_complex)); */
/*   N2 = nr; */
/*   N2P = nr; */
/*   M1 = nc; */
/*   M2 = nr; */

/*   basis_xform = c3tensor_alloc(N2,N2P,M2); */
/*   for(n2 = 0; n2 < N2; n2++) { */
    
/*     for(n2p = 0; n2p < N2P; n2p++) { */
      
/*       for(q2 = 0; q2 < M2; q2++) { */
/* 	zarg = (2.0 * pi * (n2p - n2) * (q2 - M2/2))/(double) M2; */
/* 	basis_xform[n2][n2p][q2][0] = cos(zarg)/(double) M2; */
/* 	basis_xform[n2][n2p][q2][1] = sin(zarg)/(double) M2; */
/*       } */

/*     } */

/*   } */

/*   for(sl=0; sl < ns; sl++) { */
/*     printf("starting kernel calc for sl=%d ... \n", sl); */
/*     e2 = c3tensor_alloc(N2,M2,M1); */
/*     for(n2 = 0; n2 < N2; n2++) { */
/*       tn2 = (n2 - N2/2) * Tl; */
/*       for(q2 = 0; q2 < M2; q2++) { */
      
/* 	for(q1 = 0; q1 < M1; q1++) { */
/* 	  // nr rows per slice */
/* 	  // nc pts per row */
/* 	  if(vmask[sl][q2][q1]) { */
/* 	    zarg = tn2 * fmap[sl][q2][q1]; */
/* 	    e2[n2][q2][q1][0] = cos(zarg); */
/* 	    e2[n2][q2][q1][1] = sin(zarg); */
/* 	  } else { */
/* 	    e2[n2][q2][q1][0] = 0.0; */
/* 	    e2[n2][q2][q1][1] = 0.0; */
/* 	  } */
/* 	} */
/*       } */
/*     } */
  
/*     //kern = f3tensor(M1,N2,N2P); */
/*     // k is (M1,N2,N2P) */
/*     // basis_xform is (N2,N2P,M2) */
/*     // e2 is (N2,M2,M1) */
/*     for(q1 = 0; q1 < M1; q1++) { */
/*       for(n2 = 0; n2 < N2; n2++) { */
/* 	for(n2p = 0; n2p < N2P; n2p++) { */
/* 	  sum[0][0] = 0.0; */
/* 	  sum[0][1] = 0.0; */
/* 	  for(q2 = 0; q2 < M2; q2++) { */
/* 	    if(vmask[sl][q2][q1]) { */
/* 	      sum[0][0] += (basis_xform[n2][n2p][q2][0] * e2[n2][q2][q1][0]) */
/* 		          -(basis_xform[n2][n2p][q2][1] * e2[n2][q2][q1][1]); */

/* 	      sum[0][1] += (basis_xform[n2][n2p][q2][1] * e2[n2][q2][q1][0]) */
/* 		          +(basis_xform[n2][n2p][q2][0] * e2[n2][q2][q1][1]); */
/* 	    } */
/* 	  } */
/* 	  kernel[sl][q1][n2][n2p][0] = sum[0][0]; */
/* 	  kernel[sl][q1][n2][n2p][1] = sum[0][1]; */
/* 	} */
/*       } */
/*     } */
/*   } */
/*   free(basis_xform); */
/*   free(e2); */
/*   free(sum); */
/* } */



/* zsolve_regularized takes the equation Ax = y and solves a regularized
   version (AhA + (lm^2)I)x = (Ah)y
   data shapes:
   A is MxN (usually square) (row-major)
   y is MxNRHS (row-major)
   x is NxNRHS (row-major)

   THIS could be refactored to not use as many transposes!
*/
void zsolve_regularized(fftw_complex *A, fftw_complex *y, fftw_complex *x,
			int M, int N, int NRHS, double lambda)
{
  
  fftw_complex *A2, *A2_cm, *x_cm;
  int k, l, INFO, *IPIV;
  double lm_sq[2] = {0.0, 0.0};

  lm_sq[0] = lambda*lambda;
  
  /* use A2 as the identity matrix for right now */
  A2 = (fftw_complex *) fftw_malloc((N*N) * sizeof(fftw_complex));
  A2_cm = (fftw_complex *) fftw_malloc((N*N) * sizeof(fftw_complex));
  x_cm = (fftw_complex *) fftw_malloc((NRHS*N) * sizeof(fftw_complex));
  IPIV = (int *) malloc(N * sizeof(int));
  

  bzero(A2, (N*N)*sizeof(fftw_complex));
  for(k=0; k<N; k++) A2[k + k*N][0] = 1.0;
  
  /* want to get A2 <-- (Ah)x(A) + (lm_sq)*I ... */
  /* Ah is NxM.. major stride is M (??)
     A is MxN.. major stride is N
     A2 is NxN.. major stride is N */
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
	      N, N, M, oneD,
	      (void *) A, M,
	      (void *) A, N,
	      lm_sq, (void *) A2, N);
  /* multiply data vector/matrix y by Ah, stick it in x */
  /* DOES THIS WORK WITH NRHS = 1? */
  /* Ah is NxM.. major stride is M (??)
     y is MxNRHS.. major stride is NRHS
     x is NxNRHS.. major stride is NRHS  */
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
	      N, NRHS, M, oneD,
	      (void *) A, M,
	      (void *) y, NRHS, 
	      zeroD, (void *) x, NRHS);

  /* solve the system! need to xpose ante and post to get into col-major:
     A2 is NxN, and x is NxNRHS
  */
  /* can this be done faster? */
  for(k=0; k<N; k++) {
    for(l=0; l<N; l++) {
      /* row-maj idx = k*N + l; col-maj idx = l*N + k */
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

/* This computes the (regularized) gereralized inverse of A through a 
   regularized solution: (Ah*A + lm^2*I)*C = Ah*I

   use CBLAS to get the of (Ah*A + lm^2*I) in A2,
   and then use LAPACK to solve for C. 

   Note: in column-major...
   A is conj(Ah) 
   A2 is conj(Ah*A + lm^2*I) -- since A2 is hermitian symmetric

   If conj(AhA + (lm^2)I)*C = conj(Ah), then conj(C) (the conjugate of
   the LAPACK solution, which is in col-major) is the desired solution. 
   So the final answer in row-major is the hermitian transpose of C.
   
*/
void zregularized_inverse(fftw_complex *A,
			  int M, int N, double lambda)
{
  fftw_complex *A2;
  int INFO, *IPIV, k, l, idx, idx_xp, sum = 0;
  double re, im, lmsq = lambda*lambda;

  for(k=0; k<N; k++) sum += k;
  /* memory allocating */
  A2 = (fftw_complex *) calloc((N*N), sizeof(fftw_complex));
  IPIV = (int *) malloc(N * sizeof(int));
  for(k=0; k<N; k++) {
    A2[k + k*N][0] = lmsq;
  }
  /* Ah is NxM... major stride is M (??)
     A is MxN... major stride is N
     A2 is NxN... major stride is N  */
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
	      N, N, M, oneD,
	      (void *) A, M,
	      (void *) A, N,
	      oneD, (void *) A2, N);
 
  zgesv_(&N, &M, A2, &N, IPIV, A, &N, &INFO);

  /* A is now the hermitian transpose of the desired solution MxN*/
  for(k=0; k<M; k++) {
    l = 0;
    while(l <= k) {
      idx = k*N + l;
      idx_xp = l*M + k;
      re = A[idx][0];
      im = A[idx][1];
      A[idx][0] = A[idx_xp][0];
      A[idx][1] = -A[idx_xp][1];
      A[idx_xp][0] = re;
      A[idx_xp][1] = -im;
      l++;
    }
  }

  free(A2);
  free(IPIV);
}
