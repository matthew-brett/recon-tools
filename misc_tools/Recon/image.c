#include "data.h"
#include "analyze.h"
#include "recon.h"

int get_fid_type(image_struct *img, char *base_path)
{
  char fidpath[200];
  main_hdr_struct *main_hdr;
  int nblocks, ntraces;
  FILE *fp;
  strcpy(fidpath, base_path);
  strcat(fidpath, "/fid");
  if( (fp = fopen(fidpath, "rb")) == NULL) {
    printf("error opening fid file at path %s\n", fidpath);
    exit(1);
  }
  main_hdr = (main_hdr_struct *) malloc(sizeof(main_hdr_struct));
  fread(main_hdr, sizeof(main_hdr_struct), 1, fp);
  nblocks = (int) ntohl((uint32_t) main_hdr->nblocks);
  ntraces = (int) ntohl((uint32_t) main_hdr->ntraces);
  /* 
   * known variants are:
   * "compressed" (nblocks=nvol, ntraces=nslice*npe) *
   *
   * "uncompressed" (nblocks = n_slice*n_vol, ntraces = n_pe) *
   *
   * "multislice" (nblocks = n_vol*n_pe , ntraces = n_slice) *
   */
  if(nblocks == img->n_vol_total && 
     ntraces == img->n_slice * img->n_pe) return COMPRESSED;

  if(nblocks == img->n_slice * img->n_vol_total &&
     ntraces == img->n_pe) return UNCOMPRESSED;

  if(nblocks == img->n_vol_total * img->n_pe &&
     ntraces == img->n_slice) return MULTISLICE;
  else {
    printf("unknown fid file type:\n");
    printf("\tnblocks = %d; ntraces = %d\n", nblocks, ntraces);
    printf("\tnvol(+refs) = %d; nslice = %d, npe = %d, nfe = %d\n",
	   (img->n_vol_total), img->n_slice, img->n_pe, img->n_fe);
    return -1;
  }
}   
  

/**************************************************************************
*  get_multislice_data                                                    *
*                                                                         *
*    Reads the k-space data into the memory pointed to by the data        *
*    of the image_struct.                                                 *
*                                                                         *
*  ASEMS and GEMS scans are stored in multi-slice acquisition order:      *
*  Block 1 will have n_slice traces, being the most negative PE lines     *
*  across the slices of the first volume (in slice-acq order)             *
*  So an ASEMS should have n_vol*n_pe blocks, each block should have      *
*  n_slice traces (with n_fe complex elements)                            *
*  eg, 2-vol ASEMS:                                                       *
*  block 0: n_slice rows at pe=0, vol=0                                   *
*  block 1: n_slice rows at pe=0, vol=1                                   *
*  block 2: n_slice rows at pe=1, vol=0                                   *
*  block 3: n_slice rows at pe=1, vol=1                                   *
*  ...etc...                                                              *
*                                                                         *
**************************************************************************/

 
int get_multislice_data(char *base_path, image_struct *img)
{  
  char data_path[200], str[20];
  sub_hdr_struct *sub_hdr;
  main_hdr_struct *main_hdr;
  int num, block_data_size, nblocks, ntraces, npts, ebytes, swap;
  int n_pe, n_fe, precision, n_vol, n_slice;
  int vol, pe, fe, b, t, s, n, idx;
  int re, im;
  int *acq_order;
  unsigned char *block;
  float b_real, b_imag;
  FILE *fp_data; 

  // A few assignments to make the code easier to read 
  n_pe = img->n_pe;
  n_fe = img->n_fe;
  precision = img->precision;
  n_vol = img->n_vol;
  n_slice = img->n_slice;

  sub_hdr = (sub_hdr_struct *) malloc(sizeof(sub_hdr_struct));
  main_hdr = (main_hdr_struct *) malloc(sizeof(main_hdr_struct));

  // Create a string containing the full path to the data. 
  strcpy(data_path, base_path);
  strcpy(str, "/fid");
  strcat(data_path, str);  
  printf("Reading the data file: %s \n", data_path); 
  
  if((fp_data = fopen(data_path,"rb")) == NULL){
    printf("Error opening fid file for data read.\n");
    exit(1);
  }  
  // Read main header of the data file 
  fread(main_hdr, sizeof(main_hdr_struct), 1, fp_data);
  nblocks = (int) ntohl((uint32_t) main_hdr->nblocks);
  ntraces = (int) ntohl((uint32_t) main_hdr->ntraces);
  npts = (int) ntohl((uint32_t) main_hdr->np);
  ebytes = (int) ntohl((uint32_t) main_hdr->ebytes);
  if (ebytes != main_hdr->ebytes) swap = 1;
  block_data_size = ntraces * npts;

  // Create an array which maps the spatial location of a slice, given by its
  // slice number index, to the temporal order in which it was acquired. 
  acq_order = (int *) malloc(n_slice*sizeof(int));
  n = 0;
  for(s=n_slice-1; s>-1; s-=2){
    acq_order[n++] = s;
  }
  for(s=n_slice-2; s>-1; s-=2){
    acq_order[n++] = s;
  }

  // Read the data into the memory pointed to by img->data. The slices in 
  // each volume are reordered from acquisition ordering to spatial ordering.
  // The signal bias contained in variables lvl and tlt are subtracted from
  // the signal data.
  block = (unsigned char *) malloc(block_data_size*precision); 
  for(b=0; b<nblocks; b++){
    fread(sub_hdr, sizeof(sub_hdr_struct), 1, fp_data);
    if (swap) {
      swap_bytes((unsigned char *) &(sub_hdr->lvl), sizeof(float));
      swap_bytes((unsigned char *) &(sub_hdr->tlt), sizeof(float));
    }
    fread(block, precision, block_data_size, fp_data);
    vol = b%img->n_vol_total;
    pe = b/img->n_vol_total;
    for(t=0; t<ntraces; t++){
      s = acq_order[t];
      for(fe=0; fe<n_fe; fe++){
	idx = t*npts + 2*fe;
	if (precision == 4) {
	  re = (int) ntohl( ((uint32_t *)block)[idx] );
	  im = (int) ntohl( ((uint32_t *)block)[idx+1] );
	} else {
	  re = (short) ntohs( ((uint16_t *)block)[idx] );
	  im = (short) ntohs( ((uint16_t *)block)[idx+1] );
	}
        img->data[vol][s][pe][fe][0] = (double) re - (double) sub_hdr->lvl;
	img->data[vol][s][pe][fe][1] = (double) im - (double) sub_hdr->tlt;
      }
    }
  }

  fclose(fp_data);
 
  // Free allocated resources
  free(main_hdr);
  free(sub_hdr);
  free(block);
  free(acq_order);

  printf("Finished reading the data file. \n\n");

  return 1;
}

/**************************************************************************
 * EPI data should be in either "compressed" (1 data block = 1 volume) or *
 * "uncompressed" (1 data block = 1 slice) format. In ??? case, the 
 * alternate PE lines should be reversed.
**************************************************************************/ 

int get_epibrs_data(char *base_path, image_struct *img, int filetype)
{  
  char data_path[200], ref2_path[200], str[20];
  sub_hdr_struct *sub_hdr;
  main_hdr_struct *main_hdr_data, *main_hdr_ref;
  int main_hdr_size, sub_hdr_size, expected_file_size;
  int block_data_size, swap = 0;
  int n_pe, n_fe, precision, n_vol, n_slice;
  int vol, pe, fe, b, t, s, n, idx;
  int nblocks_data, nblocks_ref, ntraces, npts, bbytes, tbytes, ebytes;
  int re, im;
  int *acq_order;
  unsigned char *block;
  float b_real, b_imag;
  fftw_complex ***data;
  FILE *fp_data, *fp_ref2, *fp; 
  

  // A few assignments to make the code easier to read 
  main_hdr_size = sizeof(main_hdr_struct);
  sub_hdr_size = sizeof(sub_hdr_struct);
  n_pe = img->n_pe;
  n_fe = img->n_fe;
  precision = img->precision;
  n_vol = img->n_vol;
  n_slice = img->n_slice;

  sub_hdr = (sub_hdr_struct *) malloc(sub_hdr_size);
  main_hdr_data = (main_hdr_struct *) malloc(main_hdr_size);
  main_hdr_ref = (main_hdr_struct *) malloc(main_hdr_size);

  // Create a string containing the full path to the data. 
  strcpy(data_path, base_path);
  strcat(data_path, "/fid");
  bzero(ref2_path, 200);
  strncpy(ref2_path, data_path, (strstr(data_path, "_data") - data_path));
  strcat(ref2_path, "_ref_2.fid/fid");
  printf("Reading the data file: %s \n", data_path); 
  
  if((fp_data = fopen(data_path,"rb")) == NULL){
    printf("Error opening fid file for data read: %s.\n", data_path);
    exit(1);
  }
  if((fp_ref2 = fopen(ref2_path, "rb")) == NULL){
    printf("Error opening 2nd ref file: %s.\n", ref2_path);
    exit(1);
  }

  fread(main_hdr_data, main_hdr_size, 1, fp_data);
  fread(main_hdr_ref, main_hdr_size, 1, fp_ref2);

  nblocks_data = (int) ntohl((uint32_t) main_hdr_data->nblocks);
  nblocks_ref = (int) ntohl((uint32_t) main_hdr_ref->nblocks);
  ntraces = (int) ntohl((uint32_t) main_hdr_data->ntraces);
  npts = (int) ntohl((uint32_t) main_hdr_data->np);
  bbytes = (int) ntohl((uint32_t) main_hdr_data->bbytes);
  tbytes = (int) ntohl((uint32_t) main_hdr_data->tbytes);
  ebytes = (int) ntohl((uint32_t) main_hdr_data->ebytes);
  if (ebytes != main_hdr_data->ebytes) swap = 1;

  if( !(filetype==COMPRESSED || filetype==UNCOMPRESSED) ) {
    printf("unknown FID file type for EPI data.\n");
    exit(1);
  }
  /* ???could check here to make sure the file length is correct */
  block_data_size = ntraces*npts;
/*   block_data_size = (filetype == COMPRESSED) ? n_slice*n_pe*n_fe*2  */
/*                                              : n_pe*n_fe*2; */

  block = (unsigned char *) malloc(block_data_size * precision);

  // Create an array which maps the spatial location of a slice, given by its
  // slice number index, to the temporal order in which it was acquired. 
  acq_order = (int *) malloc(n_slice*sizeof(int));
  n = 0;
  for(s=n_slice-1; s>-1; s-=2){
    acq_order[n++] = s;
  }
  for(s=n_slice-2; s>-1; s-=2){
    acq_order[n++] = s;
  }

  /* get ref vols out of the way */
  n = 0;
  while(n < 2) {
    if (n) {
      data = img->ref2;
      fp = fp_ref2;
    } else {
      data = img->ref1;
      fp = fp_data;
    }
    n ++;
    for(b=0 ; b<nblocks_ref; b++) {
      fread(sub_hdr, sub_hdr_size, 1, fp);
      if (swap) {
	swap_bytes((unsigned char *) &(sub_hdr->lvl), sizeof(float));
	swap_bytes((unsigned char *) &(sub_hdr->tlt), sizeof(float));
      }
      fread(block, precision, block_data_size, fp);
      for(t=0; t<ntraces; t++) {
	/* assume that n_fe = npts/2 */
	if (filetype == COMPRESSED) {
	  s = acq_order[t/n_pe];
	  pe = t%n_pe;
	} else {
	  s = acq_order[b%n_slice];
	  pe = t;
	}
	for(fe=0; fe<n_fe; fe++) {
	  idx = t*npts + 2*fe;
	  /* ntohl() and ntohs() do nothing if this machine is big-endian */
	  if (precision == 4) {
	    re = (int) ntohl( ((uint32_t *)block)[idx] );
	    im = (int) ntohl( ((uint32_t *)block)[idx+1] );
	  } else {
	    re = (short) ntohs( ((uint16_t *)block)[idx] );
	    im = (short) ntohs( ((uint16_t *)block)[idx+1] );
	  }
	  data[s][pe][fe][0] = (double) re - (double) sub_hdr->lvl;
	  data[s][pe][fe][1] = (double) im - (double) sub_hdr->tlt;
	}
      }
    }
  }
  
  /* now get data */
  for(b=nblocks_ref; b < nblocks_data; b++) {
    fread(sub_hdr, sub_hdr_size, 1, fp_data);
    if (swap) {
      swap_bytes((unsigned char *) &(sub_hdr->lvl), sizeof(float));
      swap_bytes((unsigned char *) &(sub_hdr->tlt), sizeof(float));
    }
    fread(block, precision, block_data_size, fp_data);
    for(t=0; t<ntraces; t++) {
      /* assume that n_fe = npts/2 */
      if (filetype == COMPRESSED) {
	vol = b - 1;  // remember to offset by 1 ref volume
	s = acq_order[t/n_pe];
	pe = t%n_pe;
      } else {
	vol = b/n_slice - 1;  // ditto above
	s = acq_order[b%n_slice];
	pe = t;
      }
      for(fe=0; fe<n_fe; fe++) {
	idx = t*npts + 2*fe;
	if (precision == 4) {
	  /* ntohl() and ntohs() do nothing if this machine is big-endian */
	  re = (int) ntohl( ((uint32_t *)block)[idx] );
	  im = (int) ntohl( ((uint32_t *)block)[idx+1] );
	} else {
	  re = (short) ntohs( ((uint16_t *)block)[idx] );
	  im = (short) ntohs( ((uint16_t *)block)[idx+1] );
	}
	img->data[vol][s][pe][fe][0] = (double) re - (double) sub_hdr->lvl;
	img->data[vol][s][pe][fe][1] = (double) im - (double) sub_hdr->tlt;
      }
    }
  }

  fclose(fp_data);
  fclose(fp_ref2);
  // Free allocated resources
  free(main_hdr_data);
  free(main_hdr_ref);
  free(sub_hdr);
  free(block);
  free(acq_order);

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
      image->fov = atof(key2);
      printf("fov: %f \n", image->fov);
    } 
    if(strcmp(keyword, "pss") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line,"%s ", key1);
      image->n_slice = atoi(key1);
      printf("n_slice: %d \n", image->n_slice);
//      sscanf(line,"%{%f%}", image->pss); 
    } 
    if(strcmp(keyword, "dp") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line,"%s %s", key1, key2);
      printf("key2: %s: \n", key2);
      if(strcmp(key2,"\"y\"") == 0){
        image->precision = 4;
      }else{
        image->precision = 2;
      }  
      printf("precision: %d \n", image->precision);
    } 
/*     if(strcmp(keyword, "image") == 0){ */
/*       fgets(line, sizeof(line), fp); */
/*       sscanf(line,"%s ", key1); */
/*       image->n_vol = atoi(key1); */
/*       printf("n_vol: %d \n", image->n_vol); */
/*     }  */
    if(strcmp(keyword, "pslabel") == 0){
      fgets(line, sizeof(line), fp);
      sscanf(line, "%s %s", key1, key2);
      strcpy(image->pslabel, key2);
      printf("pslabel: %s\n", image->pslabel);
    }
    if(strcmp(keyword, "asym_time")==0){
      fgets(line, sizeof(line), fp);
      sscanf(line, "%s %s %s", key1, key2, key3);
      image->asym_times[0] = atof(key2);
      image->asym_times[1] = atof(key3);
      printf("asym_times: %f, %f\n", image->asym_times[0], image->asym_times[1]);
    }
    if(strcmp(keyword, "acqcycles") == 0) {
      fgets(line, sizeof(line), fp);
      sscanf(line, "%s %s", key1, key2);
      image->n_vol_total = atoi(key2);
      printf("n_vol_total: %d\n", image->n_vol_total);
    }

  }
  /* pslabel "somestring" (with quotes), can't seem to strip that! */
  if( !strcmp(image->pslabel, "\"epidw\"") ) {
    image->n_refs = 1;
    image->n_vol = image->n_vol_total - 1;
  } else {
    image->n_vol = image->n_vol_total;
  }

  fclose(fp);
  printf("Finished reading the procpar file. \n\n"); 

  return;
}  

/**************************************************************************
 * write an analyze file *
 * analyze output types are uint8, int16, int32, float, double, and       *
 * complex float (2 adjacent floats). We'll use double or complex only.   *
 * The function xform will convert imaginary data to real data (doubles), *
 * or if it's null, we'll know to write out the complex type. The arg     *
 * iterates_on says which dimension the xform works on (eg 3 = slicewise) *
 **************************************************************************/

void write_analyze(image_struct *image, double *xform (), 
		   char *out_file, int iterates_on)
{
  header_key *hdrkey;
  image_dimension *imgdim;
  data_history *datahist;
  char hdr[220], img[220];//, image_type[20];//, out_type[20];
  int n_vol, n_slice, n_pe, n_fe, vol, slice, pe, fe, slice_sz, val_1;
  int l, m, n, chunk_sz, offset, loop_sz;
  double im1, im2, re1, re2, mag;
  double *data_chunk;
  FILE *fp;
 
  printf("Writing Analyze file to disk. \n");
  n_vol = image->n_vol;
  n_slice = image->n_slice;
  n_pe = image->n_pe;
  n_fe = image->n_fe;
  
  
  // Allocate some temporary memory.
  imgdim = (image_dimension *) calloc(1, sizeof(image_dimension));
  datahist = (data_history *) calloc(1, sizeof(datahist));
  hdrkey = (header_key *) calloc(1, sizeof(header_key));

  hdrkey->sizeof_hdr = 348;

  imgdim->dim[4] = n_vol;
  imgdim->dim[3] = n_slice;
  imgdim->dim[2] = n_pe;
  imgdim->dim[1] = n_fe;
  imgdim->dim[0] = 4;
  imgdim->datatype = (xform==NULL) ? DT_COMPLEX : DT_DOUBLE;
  imgdim->bitpix = 64;  /* this is 64 in any case, I think (double or complex) */
  imgdim->pixdim[4] = 1.0;
  imgdim->pixdim[3] = image->thk;
  imgdim->pixdim[2] = image->fov/(float)n_pe;
  imgdim->pixdim[1] = image->fov/(float)n_fe;


  strcpy(hdr,out_file);
  strcat(hdr, ".hdr");
  if ((fp = fopen(hdr,"wb")) == NULL){
    printf("Error opening analyze file. Check the path. \n");
    exit(1);
  }

  fwrite(hdrkey, sizeof(header_key), 1, fp);
  fwrite(imgdim, sizeof(image_dimension), 1, fp);
  fwrite(datahist, sizeof(data_history), 1, fp);
  fclose(fp);
  
  strcpy(img,out_file);
  strcat(img,".img");
  if ((fp = fopen(img,"wb")) == NULL){
    printf("Error opening analyze image file. Check the path. \n");
    exit(1);
  }

  if (iterates_on == 2) {
    chunk_sz = n_fe;
    loop_sz = n_vol*n_slice*n_pe;
  } else if (iterates_on == 3) {
    chunk_sz = n_pe*n_pe;
    loop_sz = n_vol*n_slice;
  } else if (iterates_on == 5) {
    chunk_sz = n_vol*n_slice*n_pe*n_fe;
    loop_sz = 1;
  } else {
    /* make this the catch-all case: volume-by-volume */
    chunk_sz = n_slice*n_pe*n_fe;
    loop_sz = n_vol;
  }
  if (xform != NULL) data_chunk = (double *) malloc(sizeof(double)*chunk_sz);
  for (l = 0; l < loop_sz; l++) {
    offset = l*chunk_sz;
    if (xform != NULL) {
      xform(data_chunk, (const fftw_complex *) ***image->data + offset, chunk_sz);
      fwrite(data_chunk, chunk_sz, sizeof(double), fp);
    } else {
      printf("skipping complex write for now!\n");
    }
  }

  fclose(fp);
  
  free(data_chunk);
  free(hdrkey);
  free(imgdim);
  free(datahist);

  printf("Finished writing Analyze file to disk. \n\n");

}
