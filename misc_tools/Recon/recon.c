/****************************************************************************
* recon.c                                                                   *
*                                                                           *
* To compile:                                                               *
*  gcc -Wall -g recon.c -o recon `pkg-config --cflags gtk+-2.0`             *
*     `pkg-config --libs gtk+-2.0`                                          *
*                                                                           *
* To run:                                                                   *
*   recon fid_dir outfile oplist                                            *
*                                                                           *
* Where fid_dir is the path to the directory containing the procpar and fid *
* data files, outfile is the user defined output file name, and oplist is   *
* the path to the file.                                                     *
*                                                                           * 
****************************************************************************/

#include "recon.h"
#include "data.h"
#include "bpc.h"

int main(int argc, char* argv[])
{

  image_struct *img;
  op_struct  *op_seq;
  char oplist_path[200], base_path[200];//, str[10]; 
  //char out_path[200];
  int n;
  double *f ();
  
  printf("\nHello from recon \n\n");

  /* Allocate memory for the op_seq struct. The members of the op_struct
  struct are assigned values in the function named read_oplist. The 
  function read_oplist reads the oplist file to determine the operations 
  the user wants to perform and the value of any extra option-specific
  parameters. When adding new operations you must make read_oplist 
  aware of them. */
  op_seq = (op_struct *) malloc(MAX_OPS * sizeof(op_struct));

  /* Allocate memory for the image structure. The members of image_struct 
  are assigned in the function read_procpar and get_data. The data within
  image_struct is used within each of the operations. It contains the raw
  data and the values of parameters specific to the data acquisition. */
  img = (image_struct *) malloc(sizeof(image_struct));
  bzero(img, sizeof(image_struct));

  if(argc < 2){
    printf("\n Error: Expecting 3 arguments to recon. \n\n");
    printf(" Usage: recon fid_dir outfile oplist \n\n");
    printf("   Where fid_dir is the path to the directory containing the\n");
    printf("   procpar and fid data files (leave off the _ref2 or _data)\n"); 
    printf("   outfile is the user defined output file name, and oplist is \n");
    printf("   the path to the oplist  file. \n\n");
    exit(0);
  }

  /* Parse the command line. */  
  strcpy(oplist_path, argv[1]);

  /* Read the oplist file. */
  read_oplist(oplist_path, op_seq);
  
/*   /\* Read procpar and put parameters in the image_struct. *\/ */
/*   read_procpar(base_path, img); */
/*   img->data = c4tensor_alloc(img->n_vol, img->n_slice, img->n_pe, img->n_fe); */
/*   img->ref1 = c3tensor_alloc(img->n_slice, img->n_pe, img->n_fe); */
/*   img->ref2 = c3tensor_alloc(img->n_slice, img->n_pe, img->n_fe); */

/*   /\* Read the k-space data into memory and assign pointers to that data. *\/ */
/*   get_epibrs_data(base_path, img); */
  
  /* Perform the operations. */
  n = 0;
  while(op_seq[n].op_active == 1){
    op_seq[n].op(img, op_seq[n]);
    printf("op name: %s\n", op_seq[n].op_name);
    n ++;
  }

  /* Release resources. */
  if(img->data) free_c4tensor(img->data);
  if(img->ref1) free_c3tensor(img->ref1);
  if(img->ref2) free_c3tensor(img->ref2);
  if(img->fmap) free_d3tensor(img->fmap);
  if(img->mask) free_d3tensor(img->mask);
  free(img);
  free(op_seq);

  return (0);
}



/**************************************************************************
*  read_oplist                                                            *
*                                                                         *
*  Read the procpar file in the fid directory.                            *
**************************************************************************/


void read_oplist(char *oplist_path, op_struct *op_seq)
{   
  char line[70];         
  int n;            
  FILE *fp;
 
  printf("Reading the oplist file %s. \n", oplist_path);

  if ((fp = fopen(oplist_path,"r")) == NULL){
    printf("Error opening oplist file. Check the path. \n");
    exit(1);
  }

  n = 0;
  while(fgets(line, sizeof(line), fp) != NULL){
    sscanf(line,"%s %s %s %s %s", op_seq[n].op_name, op_seq[n].param_1, 
           op_seq[n].param_2, op_seq[n].param_3, op_seq[n].param_4);
    //printf("OP_1 %s OP_2 %s \n", op_seq[n].op_name, op_seq[n].param_1);
    /* if this was a blank line, catch it here! */
    if (!strcmp(op_seq[n].op_name, "\0") ||
	op_seq[n].op_name[0] == '#') {
      continue;
    }
    else if(strcmp(op_seq[n].op_name,"read_image")==0){
      op_seq[n].op = read_image;
    }
    else if(strcmp(op_seq[n].op_name,"bal_phs_corr")==0){
      op_seq[n].op = bal_phs_corr;
    }
    else if(strcmp(op_seq[n].op_name,"geo_undistort")==0){
      op_seq[n].op = geo_undistort;
    }
    else if(strcmp(op_seq[n].op_name,"viewer")==0){
      op_seq[n].op = viewer;
    } 
    else if(strcmp(op_seq[n].op_name,"surf_plot")==0){
      op_seq[n].op = surf_plot;
    }
    else if(strcmp(op_seq[n].op_name,"ifft2d")==0){
      op_seq[n].op = ifft2d;
    }
    else if(strcmp(op_seq[n].op_name,"write_image")==0){
      op_seq[n].op = write_image;
    } else if(strcmp(op_seq[n].op_name,"get_fieldmap")==0){
      op_seq[n].op = get_fieldmap;
    } else {
      printf("unrecognized option: %s\n", op_seq[n].op_name);
      continue;
    }
    op_seq[n].op_active = 1; 
/*     if(strcmp(op_seq[n].op_name,"gtk_viewer")==0){ */
/*       op_seq[n].op = gtk_viewer; */
/*     }          */
    n++;
  }

  fclose(fp);
  printf("Finished reading the oplist file. \n\n");  

  return;
}    





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
*  gtk_viewer                                                             *
*                                                                         *
*  Displays the current data in a GUI viewer.                             *
**************************************************************************/

/* void gtk_viewer(image_struct *image, op_struct op_seq) */
/* { */
/*   GtkWidget *window; */
/*   GtkWidget *button; */
/*   //GtkWidget *label; */

/*   printf ("Hello from viewer. \n"); */

/*   gtk_init (NULL, NULL); */
    
/*   /\*Use gtk.image. See http://www.pygtk.org/docs/pygtk/class-gtkimage.html*\/ */
/*   window = gtk_window_new (GTK_WINDOW_TOPLEVEL); */
/*   button = gtk_button_new_with_label ("Hello World"); */

/*   g_signal_connect_swapped (G_OBJECT (button), "clicked", */
/* 			    G_CALLBACK (gtk_widget_destroy), */
/*                             G_OBJECT (window)); */
/*   gtk_container_add (GTK_CONTAINER (window), button); */
/*   gtk_widget_show (button); */
/*   gtk_widget_show (window); */

/*   gtk_main (); */
    
/*   return; */
/* } */


/**************************************************************************
 * read_image
 * reads an image from fid, path supplied in param_1
 *
 **************************************************************************/
void read_image(image_struct *image, op_struct op)
{
  int fidtype;
  read_procpar(op.param_1, image);
  fidtype = get_fid_type(image, op.param_1);
  image->data = c4tensor_alloc(image->n_vol, image->n_slice,
			       image->n_pe, image->n_fe);
  if(fidtype == MULTISLICE) {
    get_multislice_data(op.param_1, image);
    return;
  } else if(fidtype == COMPRESSED || fidtype == UNCOMPRESSED) {
    image->ref1 = c3tensor_alloc(image->n_slice, image->n_pe, image->n_fe);
    image->ref2 = c3tensor_alloc(image->n_slice, image->n_pe, image->n_fe);
    /* if <file-exists: blahblah_ref_2.fid> */
    get_epibrs_data(op.param_1, image, fidtype);
    image->n_refs++;
    /* else <only get_epi_data (no brs) > */
    return;
  } else {
    printf("not reading data!\n");
    exit(1);
  }
}

/**************************************************************************
 * write_image
 * writes an analyze file..
 * param_1 is the file name
 * param_2 is a string indicating the output type (mag, real, etc)
 * param_3 is a number indicating which dimension to loop on in the writing
 *       2 = write PE line per line
 *       3 = write slice per slice
 *       4 = write volume per volume
 *       5 = write all data at once
 **************************************************************************/
void write_image(image_struct *image, op_struct op)
{
  image_struct *im2;
  if( !strcmp(op.param_2, "mag") ) {
    write_analyze(image, mag, op.param_1, atoi(op.param_3), NULL);
    return;
  } else if( !strcmp(op.param_2, "real") ) {
    write_analyze(image, real, op.param_1, atoi(op.param_3), NULL);
    return;
  } else if( !strcmp(op.param_2, "imag") ) {
    write_analyze(image, imag, op.param_1, atoi(op.param_3), NULL);
    return;
  } else if( !strcmp(op.param_2, "angle") ) {
    write_analyze(image, angle, op.param_1, atoi(op.param_3), NULL);
    return;
  } else if( !strcmp(op.param_2, "complex") ) {
    write_analyze(image, NULL, op.param_1, atoi(op.param_3), NULL);
    return;
  }
  /* if none of the above were tried, move onto the debug stuff */
  im2 = (image_struct *) malloc(sizeof(image_struct));
  memmove(im2, image, sizeof(image_struct));
  im2->n_vol = 1;
  if( !strcmp(op.param_2, "fmap") ) {
    if (image->fmap)
      write_analyze(im2, NULL, op.param_1, atoi(op.param_3), **image->fmap);
    else
      printf("fieldmap was never computed!\n");
    return;
  } else if( !strcmp(op.param_2, "mask") ) {
    if (image->mask)
      write_analyze(im2, NULL, op.param_1, atoi(op.param_3), **image->mask);
    else
      printf("mask was never computed!\n");
    return;
  }
  printf("requested output type not recognized: %s\n", op.param_2);
}

/**************************************************************************
* bal_phs_corr                                                            *
*                                                                         *
*  An operation on k-space data                                           *
**************************************************************************/ 
/* void bal_phs_corr(image_struct *image, op_struct op_seq) */
/* {                         */

/*   printf("Hello from bal_phs_corr \n"); */
/*   printf("Number of Ref scans %d\n", image->n_refs); */

/*   /\* Multiply ref scan 1 data array elements by the complex conjugate */
/*     of ref scan 2 data array elements *\/ */

/*   /\* Calculate the angle nd divide by 2.0 *\/ */

/*   /\* Correct the data *\/ */

/*   return; */
/* }         */

/**************************************************************************
* ifft2d                                                                  *
*                                                                         *
*  An operation on k-space data                                           *
**************************************************************************/
void ifft2d(image_struct *image, op_struct op)
{
  int npe, nfe, dsize, k, slice;
  double tog = 1.0;
  fftw_complex *imspc_vec, *dp;
  fftw_plan IFT2D;

  printf("Calculating the FFT. \n");

  npe = image->n_pe;
  nfe = image->n_fe;
  dsize = npe * nfe;
  imspc_vec = (fftw_complex *) fftw_malloc(dsize * sizeof(fftw_complex));
  
  for(slice=0; slice < image->n_slice*image->n_vol; slice++) {
    dp = ***(image->data) + slice*dsize;
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
    fftw_destroy_plan(IFT2D);
    tog = 1.0;
    for(k=0; k<dsize; k++) {
      dp[k][0] = imspc_vec[k][0]*tog/(double) dsize;
      dp[k][1] = imspc_vec[k][1]*tog/(double) dsize;
      if( (k+1)%nfe ) {
	tog *= -1.0;
      }
    }
  }
  fftw_free(imspc_vec);
  printf("Finished calculating the FFT. \n\n");
}

/**************************************************************************
* get_fieldmap                                                            *
*                                                                         *
* Compute a fieldmap from an asems scan (file name given in param_1),     *
* then stick it and the associated mask into image->fmap, image->mask     *
**************************************************************************/
void get_fieldmap(image_struct *image, op_struct op)
{
  image_struct *asems;

  /* Be a little disingenuous with our operation pipelining..
     this local ref to op has the correct param_1 for read_image,
     so use it for read_image */
  asems = (image_struct *) malloc(sizeof(image_struct));
  read_image(asems, op);
  ifft2d(asems, op);
  image->fmap = d3tensor_alloc(asems->n_slice, asems->n_pe, asems->n_fe);
  image->mask = d3tensor_alloc(asems->n_slice, asems->n_pe, asems->n_fe);
  asems->fmap = image->fmap;
  asems->mask = image->mask;
  compute_field_map(asems);
  free(asems);
}

/**************************************************************************
* geo_undistort                                                           *
*                                                                         *
*  An operation on k-space data                                           *
**************************************************************************/
 
void geo_undistort(image_struct *image, op_struct op)
{                        

  printf("Hello from geo_undistort \n");
  //printf("Number of FE points %d\n", image->n_fe);

  return;
}        



/**************************************************************************
* viewer                                                                  *
*                                                                         *
* Plotting function. This function uses a pipe for interprocess           *
* communication with gnuplot                                              *
**************************************************************************/

void viewer(image_struct *image, op_struct op)
{   
  //FILE *fp1;
  FILE *fopen(), *plot2;

  printf("Hello from plotter \n");

  /* Open files which will store plot data */
  //if ((fp1 = fopen("plot11.dat","w")) == NULL){ 
  //  printf("Error can't open one or more data files\n");
  //  exit(1);
  //}
              
  plot2 = popen("/usr/bin/gnuplot> dump2", "w");
  fprintf(plot2, "%s", "set terminal x11\n");
  fflush(plot2);
  if (plot2 == NULL)
    exit(2);


  fprintf(plot2, "%s", "set title \"gray map\" \n");
  fprintf(plot2, "%s", "set pm3d map \n");
  fprintf(plot2, "%s", "set palette gray \n");
  fprintf(plot2, "%s", "set samples 100; set isosamples 100 \n");
  fprintf(plot2, "%s", "set border 4095 front linetype -1 linewidth 1.000 \n");
  fprintf(plot2, "%s", "set view map \n");
  fprintf(plot2, "%s", "set isosamples 100, 100 \n");
  fprintf(plot2, "%s", "unset surface \n");
  fprintf(plot2, "%s", "set style data pm3d \n");
  fprintf(plot2, "%s", "set style function pm3d \n");
  fprintf(plot2, "%s", "set ticslevel 0 \n");
  fprintf(plot2, "%s", "set title \"gray map\" \n");
  fprintf(plot2, "%s", "set xlabel \"x\" \n");
  fprintf(plot2, "%s", "set xrange [ -15.0000 : 15.0000 ] noreverse nowriteback \n");
  fprintf(plot2, "%s", "set ylabel \"y\" \n");
  fprintf(plot2, "%s", "set yrange [ -15.0000 : 15.0000 ] noreverse nowriteback \n");
  fprintf(plot2, "%s", "set zrange [ -0.250000 : 1.00000 ] noreverse nowriteback \n");
  fprintf(plot2, "%s", "set pm3d implicit at b \n");
  fprintf(plot2, "%s", "set palette positive nops_allcF maxcolors 0 gamma 1.5 gray \n");
  fprintf(plot2, "%s", "splot sin(sqrt(x**2+y**2))/sqrt(x**2+y**2) \n");



  /* Load files */
  //fprintf(fp1,"%f %f\n",i,y1);
       
  /* Make sure buffers flushed so that gnuplot reads up to data file */ 
  //fflush(fp1);
        
  /* Plot graph */
  fprintf(plot2, "%s", "plot 'plot11.dat' with lines \n");
  fflush(plot2);
  usleep(6000000); /* sleep for short time */
  printf("Goodbye from plotter \n");

  pclose(plot2);
  //fclose(fp1);

  return;
}


/**************************************************************************
* surf_plot                                                               *
*                                                                         *
* Plotting function. This function uses a pipe for interprocess           *
* communication with gnuplot                                              *
**************************************************************************/

void surf_plot(image_struct *image, op_struct op)
{   
  //FILE *fp1, 
  FILE *fopen(), *plot2;

  printf("Hello from plotter \n");

  /* Open files which will store plot data */
  //if ((fp1 = fopen("plot11.dat","w")) == NULL){ 
  //  printf("Error can't open one or more data files\n");
  //  exit(1);
  //}
              
  plot2 = popen("/usr/bin/gnuplot> dump2", "w");
  fprintf(plot2, "%s", "set terminal x11\n");
  fflush(plot2);
  if (plot2 == NULL)
    exit(2);

 
  fprintf(plot2, "%s", "set xlabel \"x\" \n");
  fprintf(plot2, "%s", "set ylabel \"y\" \n");
  fprintf(plot2, "%s", "set key top \n");
  fprintf(plot2, "%s", "set border 4095 \n");
  fprintf(plot2, "%s", "set xrange [-15:15] \n");
  fprintf(plot2, "%s", "set yrange [-15:15] \n");
  fprintf(plot2, "%s", "set zrange [-0.25:1] \n");
  fprintf(plot2, "%s", "set samples 25 \n");
  fprintf(plot2, "%s", "set isosamples 20 \n");

  fprintf(plot2, "%s", "set title \"Surface Plot\" \n");
  fprintf(plot2, "%s", "set pm3d; set palette \n");
  //#show pm3d
  //#show palette
  fprintf(plot2, "%s", "splot sin(sqrt(x**2+y**2))/sqrt(x**2+y**2) \n");

        
  /* Load files */
  //fprintf(fp1,"%f %f\n",i,y1);
       
  /* Make sure buffers flushed so that gnuplot reads up to data file */ 
  //fflush(fp1);
        
  /* Plot graph */
  fprintf(plot2, "%s", "plot 'plot11.dat' with lines \n");
  fflush(plot2);
  usleep(3000000); /* sleep for short time */
  printf("Goodbye from plotter \n");

  pclose(plot2);
  //fclose(fp1);

  return;
}



void time_reverse(image_struct *image, op_struct op)
{

  return;
}



void swap_bytes(unsigned char *x, int size)
{
  unsigned char c;
  unsigned short s;
  unsigned long l;

  switch (size)
  {
    case 2: // swap two bytes 
      c = *x;
      *x = *(x+1);
      *(x+1) = c;
      break;
    case 4: // swap two shorts (2-byte words) 
      s = *(unsigned short *)x;
      *(unsigned short *)x = *((unsigned short *)x + 1);
      *((unsigned short *)x + 1) = s;
      swap_bytes ((unsigned char *)x, 2);
      swap_bytes ((unsigned char *)((unsigned short *)x+1), 2);
      break;
    case 8: // swap two longs (4-bytes words) 
      l = *(unsigned long *)x;
      *(unsigned long *)x = *((unsigned long *)x + 1);
      *((unsigned long *)x + 1) = l;
      swap_bytes ((unsigned char *)x, 4);
      swap_bytes ((unsigned char *)((unsigned long *)x+1), 4);
      break;
  }
}




void compute_field_map(image_struct *img)
{
  int sl, pe, fe, n_pe, n_fe, n_slice, offset, slice_sz;
  float delta_te;
  float *pwr, *puw;
  double re, im, re1, re2, im1, im2;

  printf("Calculating the field map. \n");

  // Make some assignments for convenience.
  n_pe = img->n_pe;
  n_fe = img->n_fe;
  n_slice = img->n_slice;
  slice_sz = n_pe*n_fe;

  // Allocate memory for some temporary workspace arrays.
  pwr = (float *) malloc(sizeof(float)*slice_sz);
  puw = (float *) malloc(sizeof(float)*slice_sz);

  // Calculate the difference between the TEs of the 1st and 2nd ASEMS.
  delta_te = img->asym_times[1] - img->asym_times[0];
  //printf(" Delta TE = %f\n", delta_te);

  // At all points in the image volume calculate the phase angle of the
  // product of ASEMS image 1 times the complex conjugate of ASEMS image 2.
  for(sl=0; sl<n_slice; sl++){
    for(pe=0; pe<n_pe; pe++){
      for(fe=0; fe<n_fe; fe++){
        re1 = img->data[0][sl][pe][fe][0];
        im1 = img->data[0][sl][pe][fe][1];
        re2 = img->data[1][sl][pe][fe][0];
        im2 = img->data[1][sl][pe][fe][1];
        re = re1*re2 + im1*im2; 
        im = re1*im2 - re2*im1; 
        img->fmap[sl][pe][fe] = atan2(im, re); 
       }
    }
  }

  // Create a 3D mask indicating where the field-map SNR is sufficient.
  create_mask(img);
 
  // Unwrap the volume of phase data. 
  for(sl=0; sl<n_slice; sl++){
    // Put wrapped phase into a 1D array needed by the unwrapper function.
    for(pe=0; pe<n_pe; pe++){
      offset = pe*n_fe;
      for(fe=0; fe<n_fe; fe++){
        // Apply the mask 
        pwr[offset + fe] = (float)(img->fmap[sl][pe][fe])
                           * (float)(img->mask[sl][pe][fe]);
      }
    }
    // Unwrap a 2D slice.
    doUnwrap(pwr, puw, n_pe, n_fe);
    // Put the 1D float result in the 3D double fmap array.
    for(pe=0; pe<n_pe; pe++){
      offset = pe*n_fe;
      for(fe=0; fe<n_fe; fe++){
        img->fmap[sl][pe][fe] = (double) puw[offset + fe]/delta_te;
      }
    }
  }

  free(puw);
  free(pwr);

  printf("Finished calculating the field map. \n\n");
  
}



    
unsigned char* create_mask(image_struct *img)
{
  int sl, pe, fe, n_pe, n_fe, n_slice, slice_sz, dsize;
  int (*compar) ();
  double p02, p98, thresh, re1, re2, im1, im2;;
  double *tmp_1d;

  //printf("Entering function create_mask. \n");

  // Make some assignments for convenience.
  n_pe = img->n_pe;
  n_fe = img->n_fe;
  n_slice = img->n_slice;
  slice_sz = n_pe*n_fe;
  dsize = n_slice*slice_sz;

  // Calculate the image magnitude at all points in the spatial volume.
  for(sl=0; sl<n_slice; sl++){
    for(pe=0; pe<n_pe; pe++){
      for(fe=0; fe<n_fe; fe++){
        re1 = img->data[0][sl][pe][fe][0];
        im1 = img->data[0][sl][pe][fe][1];
        re2 = img->data[1][sl][pe][fe][0];
        im2 = img->data[1][sl][pe][fe][1];
        // Calculate the magnitude of first volume. Used in finding the mask.
        img->mask[sl][pe][fe] = sqrt(re1*re1 + im1*im1); 
      }        
    }
  }


  // Put magnitude data into a 1D array to be passed to qsort below.
  tmp_1d = (double *) malloc(dsize * sizeof(double));
  memcpy(tmp_1d, **(img->mask), dsize);

  // Find the 98th and 2th percentiles. Calculate the threshold value.
  compar = &comparator;
  qsort(tmp_1d, dsize, sizeof(double), compar);
  p02 = tmp_1d[ (int) (.02 * dsize) ]; // EXPERIMENT WITH p02
  p98 = tmp_1d[ (int) (.98 * dsize) ]; // EXPERIMENT WITH p98
  //thresh = 0.35*(p98 - p02) + p02;
  thresh = 0.1*(p98 - p02) + p02;      // EXPERIMENT WITH 0.1

  // Create the mask from the threshold value and the 3D magnitude data array.
  for(sl=0; sl<n_slice; sl++){
    for(pe=0; pe<n_pe; pe++){
      for(fe=0; fe<n_fe; fe++){
        img->mask[sl][pe][fe] = img->mask[sl][pe][fe] > thresh ? 1 : 0;
      }
    }
  }

  //printf("Leaving function create_mask. \n\n");

  //Free resources.
  free(tmp_1d);

  return 0;
}


int comparator(double *a, double *b)
{
  if (a[0] == b[0]) return 0;
  if (a[0] > b[0]) return 1;
  else return -1;
}

