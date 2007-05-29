/****************************************************************************
* auto_shim.c                                                               *
*                                                                           *
* To compile:                                                               *
*  gcc -Wall -g auto_shim.c data.c image.c -o auto_shim \                   *
*              -L./punwrap -lm -lfftw -lunwrap -llapack (?)                 *
*                                                                           *
* To run:                                                                   *
*  auto_shim  fid_dir tl outfile                                            *
*                                                                           *
* Where fid_dir is the path to the directory containing the procpar and fid *
* data files, outfile is the user defined output file name, and oplist is   *
* the path to the file.                                                     *
*                                                                           * 
****************************************************************************/

#include "auto_shim.h"
#include "data.h"
#include "punwrap/snaphu_unwrap.h"



extern int zgesdd_(char *jobz, int *m, int *n, double *a,
		       int *lda, double *s, double *u, int *ldu,
		       double *vt, int *ldvt, double *work, int *lwork,
		       double *rwork, int *iwork, int *info);


int main(int argc, char* argv[])
{

  image_struct *image;
  op_struct  *op_seq;
  char oplist_path[200], base_path[200];
  int n;
  float *fmap;
  unsigned char *vmask;

  /* Allocate memory for the image structure. The members of image_struct 
  are assigned in the function read_procpar and get_data. The data within
  image_struct is used within each of the operations. It contains the raw
  data and the values of parameters specific to the data acquisition. */
  image = (image_struct *) malloc(sizeof(image_struct));

  if(argc < 4){
    printf("\n Error: Expecting 3 arguments to recon. \n\n");
    printf(" Usage: recon fid_dir tl outfile\n\n");
    printf("   Where fid_dir is the path to the directory containing the\n");
    printf("   procpar and fid data files (leave off the _ref2 or _data)\n"); 
    printf("   tl is the time elapsed between PE sampling \n");
    exit(0);
  }

  /* Parse the command line. */
  strcpy(base_path, argv[1]); 
  
  /* Read procpar and put parameters in the image_struct. */
  read_procpar(base_path, image);

  /* Read the k-space data into memory and assign pointers to that data. */
  get_data(base_path, image);

  /* Perform the sequence of events: */
  /* fourier xform the asems         */
  /* compute the field inhomo map(s) and mask(s) */
  /* get the distortion kernels      */
  /* ????                            */

  write_analyze(image, "ksp.ana");
  ifft2d(image);
  write_analyze(image, "isp.ana");
  // fmap and vmask will both be assigned and filled here (free later)
  compute_field_map(image, &fmap, &vmask);

  image_struct *fmap_im = (image_struct *) malloc(sizeof(image_struct));
  copy_image(image, fmap_im, 0);
  int k, vsize;
  vsize = image->dsize / image->n_vol;
  float Tl = 1.0;
  for(k=0; k<image->dsize; k++) {
    if(k < vsize) {
      fmap_im->data[k][0] = (double) fmap[k];
      fmap_im->data[k][1] = 0.0;
    } else {
      fmap_im->data[k][0] = (double) vmask[k-vsize];
      fmap_im->data[k][1] = 0.0;
    }
  }
  write_analyze(fmap_im, "fmap+mask.ana");

  find_kernels(fmap, vmask, image->n_slice_vol, 
	       image->n_pe, image->n_fe, atof(argv[2]));
  

  write_analyze(image, argv[3]);
  /* Put an option to view data at certain key steps (possibly multiple)  */
  /* in each operation. Create a helper function to save a data matrix    */
  /* (real part, imaginary part, magnitude data, phase data, etc) to a    */
  /* tmp file which can be piped to a gnuplot viewer (image type, surface */
  /* plot, 2D curve). Options in the oplist files for each operation will */
  /* enable the viewers.                                                  */

  /* Release resources. */
  free(image);
  free(fmap);
  free(vmask);
  free(fmap_im);
/*   free(op_seq); */

  return (0);
}


/**************************************************************************
*  read_oplist                                                            *
*                                                                         *
*  Read the procpar file in the fid directory.                            *
**************************************************************************/


/* void read_oplist(char *oplist_path, op_struct *op_seq) */
/* {    */
/*   char line[70];          */
/*   int n;             */
/*   FILE *fp; */
 
/*   printf("Reading the oplist file %s. \n", oplist_path); */

/*   if ((fp = fopen(oplist_path,"r")) == NULL){ */
/*     printf("Error opening oplist file. Check the path. \n"); */
/*     exit(1); */
/*   } */

/*   n = 0; */
/*   while(fgets(line, sizeof(line), fp) != NULL){ */
/*     if(strcmp(line,"\n") == 0) continue; */
/*     sscanf(line,"%s %s %s %s %s", op_seq[n].op_name, op_seq[n].param_1,  */
/*            op_seq[n].param_2, op_seq[n].param_3, op_seq[n].param_4); */
/*     //printf("OP_1 %s OP_2 %s \n", op_seq[n].op_name, op_seq[n].param_1); */
/*     op_seq[n].op_active = 1; */
/*     if(strcmp(op_seq[n].op_name,"bal_phs_corr")==0){ */
/*       op_seq[n].op = bal_phs_corr; */
/*     } */
/*     if(strcmp(op_seq[n].op_name,"geo_undistort")==0){ */
/*       op_seq[n].op = geo_undistort; */
/*     } */
/*     if(strcmp(op_seq[n].op_name,"ifft2d")==0){ */
/*       op_seq[n].op = ifft2d; */
/*     } */
/*     if(strcmp(op_seq[n].op_name,"compute_field_map")==0){ */
/*       op_seq[n].op = compute_field_map; */
/*     } */
/*     n++; */
/*   } */

/*   fclose(fp); */
/*   printf("Finished reading the oplist file. \n\n");   */

/*   return; */
/* }     */


/****************************** OPERATIONS ********************************
*                                                                         *
* This section contains functions that act upon the data and replace it.  *
* To add a new operation it must be passed the the op_seq and image       *
* structs. Here is a typical function declaration for some function foo:  *
*                                                                         *
*   void foo(image_struct *image, op_struct op_seq)                       *
*                                                                         *
* This declaration must be included in the preprocessor file recon.h. In  *
* the function must be add to the code in the read_oplist function. It    *
* be obvious how to do so by simply taking a look at read_oplist.         *
*                                                                         *
****************************** OPERATIONS ********************************/


/**************************************************************************
* ifft2d                                                                  *
*                                                                         *
*  An operation on k-space data                                           *
**************************************************************************/
void ifft2d(image_struct *image)
{
  fftw_complex *imspc_vec, *dp;
  int npe, nfe, dsize, k, slice;
  fftw_plan IFT2D;
  double tog = 1.0;

  npe = image->n_pe;
  nfe = image->n_fe;
  dsize = npe * nfe;
  imspc_vec = (fftw_complex *) malloc(dsize * sizeof(fftw_complex));
  
  for(slice=0; slice < image->n_slice_vol*image->n_vol; slice++) {
    dp = image->data + slice*dsize;
    IFT2D = fftw_plan_dft_2d(nfe, npe, dp, imspc_vec, +1,
			     FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    for(k=0; k<dsize; k++) {
      dp[k][0] *= tog;
      dp[k][1] *= tog;
      if( (k+1)%nfe ) {
	tog *= -1.0;
      }
    }
    fftw_execute(IFT2D);
    tog = 1.0;
    for(k=0; k<dsize; k++) {
      dp[k][0] = imspc_vec[k][0]*tog/(double) dsize;
      dp[k][1] = imspc_vec[k][1]*tog/(double) dsize;
      if( (k+1)%nfe ) {
	tog *= -1.0;
      }
    }
  }
  free(imspc_vec);
}

void compute_field_map(image_struct *image, float **fmap, unsigned char **vmask)
{
  int vol_size, k;
  float asym_time;
  fftw_complex *conj_vol, *dvol;
  float *wr_phase, *uw_phase, *pwr, *puw;
  double *mag;
  unsigned char *mask;

  vol_size = image->dsize/image->n_vol;
  /* finding the phase diff between the 2nd and 1st acquisitions */
  asym_time = image->asym_times[1] - image->asym_times[0];
  printf("asym_time = %f\n", asym_time);

  // returns a new array into conj_vol (free later)
  conj_vol = Carray_conj(image->data, vol_size);
  printf("got conj volume ... ");

  dvol = image->data + vol_size;
  // here first argument is mutable, will contain the pointwise product
  Carray_mult(conj_vol, (const fftw_complex *) dvol, vol_size);
  printf("got diff volume ... ");
  printf("get mask ... ");

  // mag and mask are new arrays (mag free later)
  mag = Carray_mag(conj_vol, vol_size);
  // mag is actually like mag^2 since it came from a product of 2 vols
  for(k=0; k<vol_size; k++) mag[k] = sqrt(mag[k]);
  mask = mask_from_mag(mag, vol_size);
  printf("got mask \n");
  
  // wr_phase free later
  wr_phase = (float *) malloc(vol_size * sizeof(float));
  uw_phase = (float *) malloc(vol_size * sizeof(float));
  for(k=0; k<vol_size; k++) {
    wr_phase[k] = (float) Cangle(conj_vol[k]) * mask[k];
  }
  printf("got wrapped phase ... ");

  for(k=0; k<image->n_slice_vol; k++) {
    pwr = wr_phase + k*(image->n_pe * image->n_fe);
    puw = uw_phase + k*(image->n_pe * image->n_fe);
    printf("unwrapping slice %d \n",k);
    doUnwrap(pwr, puw, image->n_pe, image->n_fe);
  }
  printf("computing rad/s ... ");
  for(k=0; k<vol_size; k++) {
    uw_phase[k] = mask[k] ? uw_phase[k]/= asym_time : 0.0;
  }  
  *fmap = uw_phase;
  *vmask = mask;
  free(conj_vol);
  free(wr_phase);
  free(mag);
  printf("out of compute field map\n");
}

void find_kernels(float *fmap, unsigned char *vmask,
		  int ns, int nr, int nc, float Tl)
{
  
  int N2,N2P,M1,M2,n2,n2p,q1,q2,sl,idx;
  double re, im, zarg, tn2;
  double pi = acos(-1.0);
  double *dp, cn;
  fftw_complex *svals;
  fftw_complex ***basis_xform;
  fftw_complex ***kern;
  fftw_complex ***e2;
  fftw_complex *sum;

  sum = (fftw_complex *) malloc(sizeof(fftw_complex));
  N2 = nr;
  N2P = nr;
  M1 = nc;
  M2 = nr;

  basis_xform = f3tensor(N2,N2P,M2);
  for(n2 = 0; n2 < N2; n2++) {
    
    for(n2p = 0; n2p < N2P; n2p++) {
      
      for(q2 = 0; q2 < M2; q2++) {
	zarg = (2.0 * pi * (n2p - n2) * (q2 - M2/2))/M2;
	basis_xform[n2][n2p][q2][0] = cos(zarg)/M2;
	basis_xform[n2][n2p][q2][1] = sin(zarg)/M2;
      }

    }

  }

  for(sl=0; sl < ns; sl++) {
    printf("starting kernel calc for sl=%d ... \n", sl);
    e2 = f3tensor(N2,M2,M1);
    for(n2 = 0; n2 < N2; n2++) {
      tn2 = (n2 - N2/2) * Tl;
      for(q2 = 0; q2 < M2; q2++) {
      
	for(q1 = 0; q1 < M1; q1++) {
	  // nr rows per slice
	  // nc pts per row
	  idx = nc * (q2 + nr*sl) + q1;
	  if(vmask[idx]) {
	    zarg = tn2 * (double) fmap[idx];
	    e2[n2][q2][q1][0] = cos(zarg);
	    e2[n2][q2][q1][1] = sin(zarg);
	  } else {
	    e2[n2][q2][q1][0] = 0.0;
	    e2[n2][q2][q1][1] = 0.0;
	  }
	}
      }
    }
  
    kern = f3tensor(M1,N2,N2P);
    // k is (M1,N2,N2P)
    // basis_xform is (N2,N2P,M2)
    // e2 is (N2,M2,M1)
    printf("condition numbers via SVD for sl %d ... \n", sl);
    for(q1 = 0; q1 < M1; q1++) {
      for(n2 = 0; n2 < N2; n2++) {
	for(n2p = 0; n2p < N2P; n2p++) {
	  sum[0][0] = 0.0;
	  sum[0][1] = 0.0;
	  for(q2 = 0; q2 < M2; q2++) {
	    if(vmask[nc * (q2 + nr*sl) + q1]) {
	      sum[0][0] += (basis_xform[n2][n2p][q2][0] * e2[n2][q2][q1][0])
		          -(basis_xform[n2][n2p][q2][1] * e2[n2][q2][q1][1]);

	      sum[0][1] += (basis_xform[n2][n2p][q2][1] * e2[n2][q2][q1][0])
		          +(basis_xform[n2][n2p][q2][0] * e2[n2][q2][q1][1]);
	    }
	  }
	  kern[q1][n2][n2p][0] = sum[0][0];
	  kern[q1][n2][n2p][1] = sum[0][1];
	}
      }
      //could compute eigenvalues here -- need to point to contiguous array"
      // kern[q1] is a double **, but kern[0][0 + q1*ncol*nrow] is a double *
      // svals should be "complex double", so fftw_complex[N2] is fine
      dp = (double *) (kern[0][0] + q1*N2*N2P);
      //eigenvals(dp, (double *) svals,N2);
      printf("%1.3e ", condition(dp, N2, N2P));
      if (q1 > 0 && ! (q1 % 8) ) printf("\n");
      //qsort(svals, N2, sizeof(float), compar);
    }
    printf("\n");
    printf("got kernel\n\n\n");
  }
  free(basis_xform);
  free(e2);
  free(sum);
  free(kern);
}
/********************* Complex number operations ************************/

/* fftw_complex  Cadd(fftw_complex a, fftw_complex b) */
/* { */
/*   fftw_complex c; */
/*   c[0]=a[0]+b[0]; */
/*   c[1]=a[1]+b[1]; */
/*   return c; */
/* } */

/* fftw_complex  Csub(fftw_complex a, fftw_complex b) */
/* { */
/*   fftw_complex c; */
/*   c[0]=a[0]-b[0]; */
/*   c[1]=a[1]-b[1]; */
/*   return c; */
/* } */

/* fftw_complex  Cmul(fftw_complex a, fftw_complex b) */
/* { */
/*   fftw_complex c; */
/*   c[0]=a[0]*b[0]-a[1]*b[1]; */
/*   c[1]=a[1]*b[0]+a[0]*b[1]; */
/*   return c; */
/* } */

/* fftw_complex  Complex(double re, double im) */
/* { */
/*   fftw_complex c; */
/*   c[0]=re; */
/*   c[1]=im; */
/*   return c; */
/* } */

/* fftw_complex  Conjg(fftw_complex z) */
/* { */
/*   fftw_complex c; */
/*   c[0]=z[0]; */
/*   c[1] = -z[1]; */
/*   return c; */
/* } */

/* fftw_complex  Cdiv(fftw_complex a, fftw_complex b) */
/* { */
/*   fftw_complex c; */
/*   double r,den; */
/*   if (fabs(b[0]) >= fabs(b[1])) { */
/*     r=b[1]/b[0]; */
/*     den=b[0]+r*b[1]; */
/*     c[0]=(a[0]+r*a[1])/den; */
/*     c[1]=(a[1]-r*a[0])/den; */
/*   } else { */
/*     r=b[0]/b[1]; */
/*     den=b[1]+r*b[0]; */
/*     c[0]=(a[0]*r+a[1])/den; */
/*     c[1]=(a[1]*r-a[0])/den; */
/*   } */
/*   return c; */
/* } */

double Cabs(fftw_complex z)
{
  double x,y,ans,temp;
  x=fabs(z[0]);
  y=fabs(z[1]);
  if (x == 0.0)
    ans=y;
  else if (y == 0.0)
    ans=x;
  else if (x > y) {
    temp=y/x;
    ans=x*sqrt(1.0+temp*temp);
  } else {
    temp=x/y;
    ans=y*sqrt(1.0+temp*temp);
  }
  return ans;
}

double Cangle(fftw_complex z)
{
  return atan2(z[1], z[0]);
}

/* fftw_complex Csqrt(fftw_complex z) */
/* { */
/*   fftw_complex c; */
/*   double x,y,w,r; */
/*   if((z[0] == 0.0) && (z[1] == 0.0)) { */
/*     c[0]=0.0; */
/*     c[1]=0.0; */
/*     return c; */
/*   } else { */
/*     x=fabs(z[0]); */
/*     y=fabs(z[1]); */
/*     if (x >= y) { */
/*       r=y/x; */
/*       w=sqrt(x)*sqrt(0.5*(1.0+sqrt(1.0+r*r))); */
/*     } else { */
/*       r=x/y; */
/*       w=sqrt(y)*sqrt(0.5*(r+sqrt(1.0+r*r))); */
/*     } */
/*     if (z[0] >= 0.0) { */
/*       c[0]=w; */
/*       c[1]=z[1]/(2.0*w); */
/*     } else { */
/*       c[1]=(z[1] >= 0) ? w : -w; */
/*       c[0]=z[1]/(2.0*c[1]); */
/*     } */
/*     return c; */
/*   } */
/* } */

/* fftw_complex  RCmul(double x, fftw_complex a) */
/* { */
/*   fftw_complex c; */
/*   c[0]=x*a[0]; */
/*   c[1]=x*a[1]; */
/*   return c; */
/* } */



/*************************** Helper functions ******************************/

// swap using char pointers
float swap_float(float d)
{
    float a;
    unsigned char *dst = (unsigned char *)&a;
    unsigned char *src = (unsigned char *)&d;

    dst[0] = src[3];
    dst[1] = src[2];
    dst[2] = src[1];
    dst[3] = src[0];
    return a;
}

/* void swap_bytes(unsigned char *a, int nbytes) */
/* { */
/*   int k; */
/*   unsigned char tmp; */
/*   for(k=0; k<nbytes/2; k++) { */
/*     tmp = a[nbytes-k-1]; */
/*     a[nbytes-k-1] = a[k]; */
/*     a[k] = tmp; */
/*   } */
/* } */

void swap_bytes(unsigned char *x, int size)
{
  unsigned char c;
  unsigned short s;
  unsigned long l;

  switch (size)
  {
    case 2: /* swap two bytes */
      c = *x;
      *x = *(x+1);
      *(x+1) = c;
      break;
    case 4: /* swap two shorts (2-byte words) */
      s = *(unsigned short *)x;
      *(unsigned short *)x = *((unsigned short *)x + 1);
      *((unsigned short *)x + 1) = s;
      swap_bytes ((char *)x, 2);
      swap_bytes ((char *)((unsigned short *)x+1), 2);
      break;
    case 8: /* swap two longs (4-bytes words) */
      l = *(unsigned long *)x;
      *(unsigned long *)x = *((unsigned long *)x + 1);
      *((unsigned long *)x + 1) = l;
      swap_bytes ((char *)x, 4);
      swap_bytes ((char *)((unsigned long *)x+1), 4);
      break;
  }
}

fftw_complex* Carray_conj(fftw_complex *zarray, const int dsize)
{
  int k;
  fftw_complex *conj_array;
  conj_array = (fftw_complex *) malloc(dsize * sizeof(fftw_complex));
  for(k=0; k<dsize; k++) {
    conj_array[k][0] = zarray[k][0];
    conj_array[k][1] = -zarray[k][1];
  }
  return conj_array;
}

void Carray_mult(fftw_complex *za1, const fftw_complex *za2, const int dsize)
{
  int k;
  double re, im;
  for(k=0; k<dsize; k++) {
    re = za1[k][0] * za2[k][0] - za1[k][1] * za2[k][1];
    im = za1[k][1] * za2[k][0] + za1[k][0] * za2[k][1];
    za1[k][0] = re;
    za1[k][1] = im;
  }
}

double* Carray_real(fftw_complex *zarray, const int dsize)
{
  double *real;
  int k;
  real = (double *) malloc(dsize * sizeof(double));
  for(k=0; k<dsize; k++) real[k] = zarray[k][0];
  return real;
}

double* Carray_imag(fftw_complex *zarray, const int dsize)
{
  double *imag;
  int k;
  imag = (double *) malloc(dsize * sizeof(double));
  for(k=0; k<dsize; k++) imag[k] = zarray[k][1];
  return imag;
}

double* Carray_mag(fftw_complex *zarray, const int dsize)
{
  double *mag;
  int k;
  mag = (double *) malloc(dsize * sizeof(double));
  for(k=0; k<dsize; k++) mag[k] = Cabs(zarray[k]);
  return mag;
}
    
unsigned char* mask_from_mag(double *mag, const int dsize)
{
  int k;
  double p02, p98, thresh;
  double *mag_copy;
  unsigned char *mask;
  int (*compar) ();
  mag_copy = (double *) malloc(dsize * sizeof(double));
  memmove(mag_copy, mag, dsize*sizeof(double));
  compar = &comparator;
  qsort(mag_copy, dsize, sizeof(double), compar);
  mask = (unsigned char *) malloc(dsize * sizeof(unsigned char));
  p02 = mag_copy[ (int) (.02 * dsize) ];
  p98 = mag_copy[ (int) (.98 * dsize) ];
  thresh = 0.1*(p98 - p02) + p02;
  for(k=0; k<dsize; k++) {
    mask[k] = mag[k] > thresh ? 1 : 0;
  }
  free(mag_copy);
  return mask;
}

int comparator(double *a, double *b)
{
  if (a[0] == b[0]) return 0;
  if (a[0] > b[0]) return 1;
  else return -1;
}

/* void eigenvals(double *a, double *e, int M) { */
/*   char JOBVL = 'N'; */
/*   char JOBVR = 'N'; */
/*   double *vl, *vr, *work, *rwork; */
/*   int N, LDA, LDVL, LDVR, LWORK, INFO; */
  
/*   N = LDA = LDVL = LDVR = M; */
/*   LWORK = 16 * N; */
/*   work = (double *) malloc(sizeof(double) * 2 * LWORK); */
/*   rwork = (double *) malloc(sizeof(double) * 2 * N); */
/*   zgeev_(&JOBVL, &JOBVR, &N, a, &LDA, e, vl, &LDVL, vr, &LDVR, */
/* 	 work, &LWORK, rwork, &INFO); */
/*   free(work); */
/*   free(rwork); */
/* } */

double condition(double *a, int M, int N) {
  char JOBZ = 'N';
  int LDA, LDU, LDVT, LWORK, IWORK, LRWORK, INFO, ns;
  double *s, *u, *vt, *work, *rwork, cond;

  LDA = LDU = M;
  LDVT = N;
  LWORK = M > N ? 2*N + M : 2*M + N;
  IWORK = M > N ? 8*N : 8*M;
  LRWORK = M > N ? 5*N : 5*M;
  ns = M > N ? N : M;
  work = (double *) malloc(2 * LWORK * sizeof(double));
  rwork = (double *) malloc(2 * LRWORK * sizeof(double));
  s = (double *) malloc(sizeof(double)*ns);
  zgesdd_(&JOBZ, &M, &N, a, &LDA, s, u, &LDU, vt, &LDVT,
	  work, &LWORK, rwork, &IWORK, &INFO);
  if (INFO==0 && s[ns-1] > 0.0) {
    cond = s[0]/s[ns-1];
  } else {
    cond = -1.0;
  }
  free(s);
  free(work);
  free(rwork);
  return cond;
}
