#include <fftw3.h>
#include <vecLib/cblas.h>

void bal_phs_corr(image_struct *image, op_struct op);
void apply_phase_correction(fftw_complex *data, fftw_complex *corrector,
			    int rowsize, int volsize, int nvols);
void unwrap_ref_volume(double *uphase, fftw_complex ***vol, 
		       int zdim, int ydim, int xdim, int xstart, int xstop);
double array_max(double *array, int len, int *max_idx);
void reverse_fe(fftw_complex *z, int n_fe, int len_z);
void fft1d(fftw_complex *zin, fftw_complex *zout, 
	   int len_xform, int len_z, int direction);
double var(double *points, int npts);
void svd_solve(double *A, double *y, double *x, int M, int N);
void linReg(double *y, double *x, double *sigma, int len, 
	    double *m, double *b, double *res);
void maskbyfit(double *line, double *sigma, double *mask, double tol, 
	       double tol_growth, int len);

