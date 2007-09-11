#include "InOutFiles.h"

void read_data(char *inputfile,float *Data, int length)
{
/* 	printf("Reading the Wrapped Values form Binary File.............>"); */
	FILE *ifptr;
	ifptr = fopen(inputfile,"rb");
	if(ifptr == NULL) printf("Error opening the file\n");
	fread(Data,sizeof(float),length,ifptr);
	fclose(ifptr);
/* 	printf(" Done.\n"); */
}

void write_data(char *outputfile,float *Data,int length)
{
/* 	printf("Writing the Wrapped Values to Binary File.............>"); */
	FILE *ifptr;
	ifptr = fopen(outputfile,"wb");
	if(ifptr == NULL) printf("Error opening the file\n");
	fwrite(Data,sizeof(float),length,ifptr);
	fclose(ifptr);
/* 	printf(" Done.\n"); */
}
