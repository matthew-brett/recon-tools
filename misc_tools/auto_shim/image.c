#include "data.h"
#include "analyze.h"
#include "auto_shim.h"
/**************************************************************************
*  get_data                                                               *
*                                                                         *
*    Reads the k-space data into previously allocated memory.             *
*                                                                         *
**************************************************************************/
 
int get_data(char *base_path, image_struct *image)
{  
  char data_path[200], ref2_path[200], str[20];
  char main_hdr[32];
  sub_hdr_struct *sub_hdr;
  int main_hdr_size, sub_hdr_size, expected_file_size;
  int num, slice_size, block_data_size, expected_data_size, nblocks, ntraces;
  int n_pe, n_fe, precision, n_vol, n_refs, n_slice_vol, n_slice_total;
  int vol, pe, fe, b, t, s, v ,n, off, t_size;
  int re, im;
  int *block, *acq_order;
  float b_real, b_imag;
  FILE *fp_data, *fp_ref2; 

  /* A few assignments to make the code easier to read */
  main_hdr_size = 32;
  sub_hdr_size = 28;
  n_pe = image->n_pe;
  n_fe = image->n_fe;
  precision = image->precision;
  n_vol = image->n_vol;
  n_refs = image->n_refs;
  n_slice_vol = image->n_slice_vol;
  n_slice_total = n_vol * n_slice_vol;

  sub_hdr = (sub_hdr_struct *) malloc(sizeof(sub_hdr_struct));

  /* Concatenate the base_path with strings that give the full path to data */
  /* and ref scan files.                                                    */
  strcpy(data_path, base_path);
  strcpy(str, "/fid");
  strcat(data_path, str);  
  printf("Reading the data file: %s \n", data_path); 
  
  /* ASEMS scans are stored in multi-slice acquisition order:          */
  /* Block 1 will have n_slice traces, being the most negative PE lines*/
  /* across the slices of the first volume (in slice-acq order)        */
  /* So an ASEMS should have n_vol*n_pe blocks, each block should have */
  /* n_slice traces (with n_fe complex elements)                       */
  /* eg, 2-vol ASEMS:                                                  */
  /* block 0: n_slice rows at pe=0, vol=0                              */
  /* block 1: n_slice rows at pe=0, vol=1                              */
  /* block 2: n_slice rows at pe=1, vol=0                              */
  /* block 3: n_slice rows at pe=1, vol=1                              */
  /* ...etc... */

  /* Compute the expected size (number of complex valued entries) of the */
  /* main data file.                                                     */
  ntraces = n_slice_vol;
  block_data_size = ntraces*n_fe*2;
  t_size = n_fe * 2;
  /* Compute the expected size of the main data file in bytes. Assumes each */
  /* complex entry is 4 bytes (two shorts: one for real part and one for    */
  /* imaginary part of entry).                                              */
  nblocks = n_vol*n_pe;
  expected_file_size = main_hdr_size 
                       + nblocks*(block_data_size*precision + sub_hdr_size);
  
  /* Open the actual FID file to determine its length */
  if((fp_data = fopen(data_path,"rb")) == NULL){
    printf("Error opening fid file for data read.\n");
    exit(1);
  }  

  /* Allocate temporary memory for storing short ints from the fid files */  
  /* The real part comes first followed by the imaginary part. The data  */ 
  /* is stored a PE line at a time.                                      */

  block = (int *) malloc(block_data_size*precision);
  acq_order = (int *) malloc(n_slice_total*sizeof(int));
  n = 0;
  for(v=0; v<n_vol; v++) {
    off = v*n_slice_vol;
    for(s=n_slice_vol-1+off; s>-1+off; s-=2){
      acq_order[n++] = s;
    }
    for(s=n_slice_vol-2+off; s>-1+off; s-=2){
      acq_order[n++] = s;
    }
  }
  /* Allocate float matrices data_r, data_i, ref1_r, ref_1i,ref2_r */
  /* and ref_2i                                                    */
  image->data = zarray(image->dsize);

  /* Read main header of the main data file */
  fread(main_hdr, sizeof(char), main_hdr_size, fp_data);
  
  for(b=0; b<nblocks; b++){
    /* Read a block of ASEMS data, which is n_slice_vol traces */
    fread(sub_hdr, sizeof(sub_hdr_struct), 1, fp_data);
    //b_real = swap_float(sub_hdr->lvl);
    //b_imag = swap_float(sub_hdr->tlt);
    swap_bytes((unsigned char *) &(sub_hdr->lvl), sizeof(float));
    swap_bytes((unsigned char *) &(sub_hdr->tlt), sizeof(float));
    b_real = sub_hdr->lvl;
    b_imag = sub_hdr->tlt;
    num = fread(block, precision, block_data_size, fp_data);
    vol = b%2;
    pe = b/2;
    for(t=0; t<ntraces; t++){
      s = acq_order[t + n_slice_vol*vol];
      /* need to get (s,pe,:) row where strides are */
      /* n_slice_total, n_pe */
      for(fe=0; fe<n_fe; fe++){
	if (precision == 4) {
	  re = (int) ntohl((uint32_t) block[t*t_size + 2*fe]);
	  im = (int) ntohl((uint32_t) block[t*t_size + 2*fe + 1]);
	} else {
	  re = (int) ntohs((uint16_t) block[t*t_size + 2*fe]);
	  im = (int) ntohs((uint16_t) block[t*t_size + 2*fe + 1]);
	}
	/* there are n_pe rows per slice */
	/* and there are n_fe pts per row */
	image->data[n_fe*(n_pe*s + pe) + fe][0] = (double) re - (double) b_real;
	image->data[n_fe*(n_pe*s + pe) + fe][1] = (double) im - (double) b_imag;
      }
    }
  }

  //view_data();

  fclose(fp_data);
  printf("Finished reading the data file. \n\n");

  return 1;
}


/**************************************************************************
*  read_procpar                                                           *
*                                                                         *
*  Read the procpar file in the fid directory.                            *
**************************************************************************/
void read_procpar(char *base_path, image_struct *image)
{                        
  char keyword[400], line[400], procpar_path[200], str[20];   
  char key1[200], key2[200], key3[200];
  FILE *fp;
 
  strcpy(procpar_path, base_path);
  strcpy(str, "/procpar");
  strcat(procpar_path, str);

  printf("Reading the procpar file %s. \n", procpar_path);  

  if ((fp = fopen(procpar_path,"r")) == NULL){
    printf("Error opening procpar file. Check the path. \n");
    exit(1);
  }

  while(fgets(line, sizeof(line), fp) != NULL){
    sscanf(line,"%s ", keyword);
    if(strcmp(keyword, "thk") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line,"%s %s", key1, key2);
      image->thk = atof(key2);
      printf("thk: %f. \n", image->thk);
    } 
    if(strcmp(keyword, "nv") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line,"%s %s", key1, key2);
      image->navs_per_seg = atoi(key2) % 32;
      image->n_pe = atoi(key2) - image->navs_per_seg;
      printf("n_pe: %d \n", image->n_pe);
      printf("navs_per_seg: %d \n", image->navs_per_seg);
    } 
    if(strcmp(keyword, "np") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line,"%s %s", key1, key2);
      image->n_fe = atoi(key2)/2;
      printf("n_fe: %d \n", image->n_fe);
    } 
    if(strcmp(keyword, "lro") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line,"%s %s", key1, key2);
      image->fov = atof(key2)*10.0;
      printf("fov: %f mm\n", image->fov);
    } 
    if(strcmp(keyword, "pss") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line,"%s ", key1);
      image->n_slice_vol = atoi(key1);
      printf("n_slice_vol: %d \n", image->n_slice_vol);
//      sscanf(line,"%{%f%}", image->pss); 
    } 
    if(strcmp(keyword, "dp") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line,"%s %s", key1, key2);
      printf("double precision? %s: \n", key2);
      if(strcmp(key2,"\"y\"") == 0){
        image->precision = 4;
      }else{
        image->precision = 2;
      }  
      printf("precision: %d \n", image->precision);
    } 
/*     if(strcmp(keyword, "cntr") == 0){ */
/*       fgets(line, sizeof(line), fp); */
/*       sscanf(line,"%s ", key1); */
/*       image->n_vol = atoi(key1); */
/*       printf("n_vol: %d \n", image->n_vol); */
/*     }  */
    if(strcmp(keyword, "asym_time")==0){
      fgets(line, sizeof(line), fp);
      sscanf(line, "%s %s %s", key1, key2, key3);
      image->asym_times[0] = atof(key2);
      image->asym_times[1] = atof(key3);
      printf("asym_times: %f, %f\n", image->asym_times[0], image->asym_times[1]);
    }
  }
  image->n_refs = 1;
  image->n_vol = 2;
  image->n_slice_total = image->n_vol * image->n_slice_vol;
  image->dsize = image->n_slice_total * image->n_pe * image->n_fe;
  fclose(fp);
  printf("Finished reading the procpar file. \n\n"); 

  return;
}  

/**************************************************************************
 * write an analyze file *
 **************************************************************************/

void write_analyze(image_struct *image, char *fname)
{
  header_key *hdrkey;
  image_dimension *imgdim;
  data_history *datahist;
  double *img_mag;
  int nvol, nslice, npe, nfe, dsize, k;
  FILE *fp;
  char hdr[200], img[200];
  nvol = image->n_vol;
  nslice = image->n_slice_vol;
  npe = image->n_pe;
  nfe = image->n_fe;
  

  hdrkey = (header_key *) calloc(1, sizeof(header_key));
  hdrkey->sizeof_hdr = 348;
  
  imgdim = (image_dimension *) calloc(1, sizeof(image_dimension));
  imgdim->dim[4] = nvol;
  imgdim->dim[3] = nslice;
  imgdim->dim[2] = npe;
  imgdim->dim[1] = nfe;
  imgdim->dim[0] = 4;
  imgdim->datatype = DT_DOUBLE;
  imgdim->bitpix = sizeof(double)*8;
  imgdim->pixdim[4] = 1.0;
  imgdim->pixdim[3] = image->thk;
  imgdim->pixdim[2] = image->fov/(float)npe;
  imgdim->pixdim[1] = image->fov/(float)nfe;

  datahist = (data_history *) calloc(1, sizeof(datahist));

  
  dsize = image->dsize;
  img_mag = Carray_real(image->data, dsize);
/*   img_mag = (double *) malloc(dsize * sizeof(double)); */
/*   for(k=0; k<dsize; k++) { */
/*     img_mag[k] = Cabs(image->data[k]); */
/*   } */

  strcpy(hdr,fname);
  strcat(hdr, ".hdr");
  if ((fp = fopen(hdr,"wb")) == NULL){
    printf("Error opening analyze file. Check the path. \n");
    exit(1);
  }
  
  fwrite(hdrkey, sizeof(header_key), 1, fp);
  fwrite(imgdim, sizeof(image_dimension), 1, fp);
  fwrite(datahist, sizeof(data_history), 1, fp);
  fclose(fp);
  free(hdrkey);
  free(imgdim);
  free(datahist);
  
  strcpy(img,fname);
  strcat(img,".img");
  if ((fp = fopen(img,"wb")) == NULL){
    printf("Error opening analyze image file. Check the path. \n");
    exit(1);
  }
  fwrite(img_mag, sizeof(double), dsize, fp);
  fclose(fp);
  free(img_mag);
}

/****************************************************************************
 * copy_image takes an existing image src and an unassigned pointer dest    *
 * and copies the attributes of src into dest, and possibily the data too   *
 ****************************************************************************/
void copy_image(const image_struct *src, image_struct *dest, int copyarray)
{
  memmove(dest, src, sizeof(image_struct));
  dest->data = zarray(dest->dsize);
  if(copyarray) {
    memmove(dest->data, src->data, src->dsize*sizeof(fftw_complex));
  }
}
