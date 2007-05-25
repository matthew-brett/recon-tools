#include <fftw3.h>

typedef struct{
  short int  scale;     // spare short word
  short int  status;       // status word for block header
  short int  index;     // spare short word
  short int  mode;     // spare short word
  long int   ctcount;     // spare long word
  float      lpval1;       // 2D-f2 left phase
  float      rpval1;       // 2D-f2 right phase
  float      lvl;     // spare float word
  float      tlt;     // spare float word
} sub_hdr_struct;


/***** The following structure contains all image data and parameters *****/

typedef struct{
  /* From procpar file */
  int n_fe;          /* Number of frequency-encoding lines */
  int n_pe;          /* Number of phase-encoding lines */
  int precision;     /* Bit-depth of the data */
  int n_refs;        /* Number of ref scans */
  float thk;         /* Slice thickness */
  int n_segs;        /* Number of segments */
  int navs_per_seg;  
  int n_slice_total; /* Number of slices (2D data) in all volumes */
  int n_slice_vol;   /* Number of slices (2D data) in a volume */
  int n_vol;         /* Number of data volumes not including the ref scans */
  float fov;         /* Field of View */
  float *pss;
  float asym_times[2];
  int dsize;         /* length of the data array, for convenience */
  /* FFTW makes transform "plans" that are reusable transformations */
/*   fftw_plan fft2d; */
/*   fftw_plan ifft2d; */

  /* From data files */
  fftw_complex *data;   /* The kspace complex data */
  /*fftw_complex *ispdata;   /* The imspace complex data */
/*   fftw_complex *ref1_r;   /\* The real part of ref1 *\/ */
/*   fftw_complex *ref1_i;   /\* The imaginary part of ref1 *\/ */
/*   fftw_complex *ref2_r;   /\* The real part of ref2 *\/ */
/*   fftw_complex *ref2_i;   /\* The imaginary part of ref2 *\/ */
} image_struct;


/* Declaration of Data IO functions */
void read_procpar(char *procpar_path, image_struct *image);
int get_data(char *data_path, image_struct *image);
void write_analyze(image_struct *image, char *fname);
void copy_image(const image_struct *src, image_struct *dest, int copyarray);
