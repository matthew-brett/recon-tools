/*****************************************************************************
* Header file for recon_main.c                                               *
*****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <string.h>
#include <fftw3.h>
#include <netinet/in.h>
#include <errno.h>
#include <vecLib/clapack.h>

//#ifndef __imageH__
#include "image.h"
//#define __imageH__
//#endif

#define   MAX_OPS                    100 

/* Define a short int complex type */
typedef struct{
   short r;                // real part
   short i;                // imaginary part
} scomplex;

/* Define a float complex type */
typedef struct{
   float r;                // real part
   float i;                // imaginary part
} fcomplex;

/* The following structure contains an operation to be performed and its 
   optional parameters. */
typedef struct{
  void (*op) ();
  char op_name[30];
  char param_1[20];
  char param_2[20];
  char param_3[20];
  char param_4[20];
  int  op_active;
} op_struct;

/* Declaration of operation functions */
void ifft2d(image_struct *image);
void compute_field_map(image_struct *image, float **fmap, unsigned char **vmask);
void find_kernels(float *fmap, unsigned char *vmask, 
		  int ns, int nr, int nc, float Tl);

/* Declaration of Data IO functions */
void read_oplist(char *oplist_path, op_struct *op_seq);

/* Declaration of helper functions */
void time_reverse(image_struct *image);
float swap_float(float d);
fftw_complex* Carray_conj(fftw_complex *zarray, const int dsize);
void Carray_mult(fftw_complex *za1, const fftw_complex *za2, const int dsize);
double* Carray_real(fftw_complex *zarray, const int dsize);
float* Carray_realsp(fftw_complex *zarray, const int dsize);
double* Carray_imag(fftw_complex *zarray, const int dsize);
double* Carray_mag(fftw_complex *zarray, const int dsize);
unsigned char* mask_from_mag(double *mag, const int dsize);
int comparator(double *a, double *b);
void eigenvals(double *a, double *e, int M);
double condition(double *a, int M, int N);


/* Declaration of complex number operations */

//Returns the complex sum of two comp
/* fftw_complex Cadd(fftw_complex a, fftw_complex b); */

/* //Returns the complex difference of two complex numbers. */
/* fftw_complex Csub(fftw_complex a, fftw_complex b); */

/* //Returns the complex product of two complex numbers. */
/* fftw_complex Cmul(fftw_complex a, fftw_complex b); */

/* //Returns the complex quotient of two complex numbers. */
/* fftw_complex Cdiv(fftw_complex a, fftw_complex b); */

/* //Returns the complex square root of a complex number. */
/* fftw_complex Csqrt(fftw_complex z); */

/* //Returns the complex conjugate of a complex number. */
/* fftw_complex Conjg(fftw_complex z); */

//Returns the absolute value (modulus) of a complex number.
double Cabs(fftw_complex z);

//Returns the angle of the complex number
double Cangle(fftw_complex z);

/* //Returns a complex number with specified real and imaginary parts. */
/* fftw_complex Complex(double re, double im); */

/* //Returns the complex product of a real number and a complex number. */
/* fftw_complex RCmul(double x, fftw_complex a); */
