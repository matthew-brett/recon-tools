#include "SecondDiff_QM.h"
#include "pi.h"

void SndDiff26(float *inData,float *outData,int x, int y, int z)
{
	int frame, row;
	int i,j,k,index;
	float me,Horizantal, Vertical, Normal, Diag1, Diag2, Diag3, Diag4, Diag5, Diag6, Diag7, Diag8, Diag9, Diag10;
	frame = x*y;
	row = x;
	for ( k=1; k<z-1; k++)
	{
		for(j=1; j<y-1; j++)
		{
			for(i=1; i<x-1; i++)
			{
				index = k*frame + j*row + i;
				
					me = inData[index];
					Horizantal = reliability(me,inData[index-1],inData[index+1]);
					Vertical = reliability(me,inData[index-row], inData[index+row]);
					Normal = reliability(me,inData[index-frame], inData[index+frame]);
					Diag1 = reliability(me,inData[index-row-1], inData[index+row+1]);
					Diag2 = reliability(me,inData[index-row+1], inData[index+row-1]);
					Diag3 = reliability(me,inData[index-frame-row-1], inData[index+frame+row+1]);
					Diag4 = reliability(me,inData[index-frame-row], inData[index+frame+row]);
					Diag5 = reliability(me,inData[index-frame-row+1], inData[index+frame+row-1]);
					Diag6 = reliability(me,inData[index-frame-1], inData[index+frame+1]);
					Diag7 = reliability(me,inData[index-frame+1], inData[index+frame-1]);
					Diag8 = reliability(me,inData[index-frame+row-1], inData[index+frame-row+1]);
					Diag9 = reliability(me,inData[index-frame+row], inData[index+frame-row]);
					Diag10= reliability(me,inData[index-frame+row+1], inData[index+frame-row-1]);
					outData[index] = pow(pow((Horizantal),2)+pow((Vertical),2)+pow((Normal),2)+pow((Diag1),2)+pow((Diag2),2)
									+pow((Diag3),2)+pow((Diag4),2)+pow((Diag5),2)+pow((Diag6),2)+pow((Diag7),2)+pow((Diag8),2)
									+pow((Diag9),2)+pow((Diag10),2),0.5);
			}
		}
	}
}


void SndDiff6(float *inData,float *outData,int x, int y, int z)
{
	int frame, row;
	int i,j,k,index;
	float me,Horizantal, Vertical, Normal;
	frame = x*y;
	row = x;
	for ( k=1; k<z-1; k++)
	{
		for(j=1; j<y-1; j++)
		{
			for(i=1; i<x-1; i++)
			{
				index = k*frame + j*row + i;
				
					me = inData[index];
					Horizantal = reliability(me,inData[index-1],inData[index+1]);
					Vertical = reliability(me,inData[index-row], inData[index+row]);
					Normal = reliability(me,inData[index-frame], inData[index+frame]);
					outData[index] = pow(pow((Horizantal),2)+pow((Vertical),2)+pow((Normal),2),0.5);
			}
		}
	}
}


float reliability(float me, float a, float b)
{
	float first_term, second_term, result;
	first_term = a - me;
	if (first_term > PI )
	{
		first_term = first_term - TWOPI;
	}
	else if (first_term < -PI )
	{
		first_term = first_term + TWOPI;
	}
	second_term = me - b;
	if (second_term > PI )
	{
		second_term = second_term - TWOPI;
	}
	else if (second_term < -PI )
	{
		second_term += TWOPI;
	}
	result = first_term - second_term;
	return result;
}
