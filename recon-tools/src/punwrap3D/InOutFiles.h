#ifndef __INOUTFILES
#define __INOUTFILES

#include <stdio.h>
#include <stdlib.h>

void read_data(char *inputfile,float *Wrapped_Volume, int total_no_voxles);
void write_data(char *outputfile,float *Data,int length);

#endif
