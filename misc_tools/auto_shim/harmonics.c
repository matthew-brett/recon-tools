/****************************************************************************
* harmonics.c                                                               *
*                                                                           *
* To compile:                                                               *
*  gcc -Wall harmonics.c -o harmonics -lm                                   *
*                                                                           *
* To run:                                                                   *
*   harmonics                                                               *
****************************************************************************/

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


/* Declaration of functions */
double plgndr(int l, int m, float x);
void surf_plot();
double shim_field(int l, int m, float x, float y, float z);

int main(int argc, char* argv[])
{
  int l, m, nx, ny, nz, n_pts;
  double x, y, z, field_val;
 
  n_pts = 3;
  l = 3;
  m = -3;

  /* Calculate (l,m) shim field at (x,y,z) and write to stndout*/
  for(nx=0;nx<n_pts;nx++){
    x=nx;
    for(ny=0;ny<n_pts;ny++){
      y=ny;
      for(nz=0;nz<n_pts;nz++){
        z=nz;
        field_val= shim_field(l,m,x,y,z);
        printf("(x,y,z): %f %f %f field_val %f\n", x, y, z, field_val);
      }
    }
  }

  return (0);
}


/****************************************************************************
* shimfield                                                                 *
*                                                                           *
* This function calculates the "standard" shim fields (having one component *
* in the z-direction only) at the given cartesian coordinates. The shim     *
* fields are indexed (and named) as follows:                                *
*                                                                           *
*   l     m               function              "common" name               *
*  ---   ---             ----------             --------------              *
*   1     0                  z                        Z                     *
*   1     1                  y                        Y                     *
*   1    -1                  x                        X                     *
*   2     0          z^2 - (x^2 + y^2)/2             Z2                     *
*   2     1                 zy                       ZY                     *
*   2    -1                 zx                       ZX                     *
*   2     2              x^2 - y^2                  X2Y2                    *
*   2    -2                 xy                       XY                     *
*   3     0        z^3 - 3z(x^2 + y^2)/2             Z3                     *
*   3     1        z^2 x - x(x^2 + y^2)/4           Z2X                     *
*   3    -1        z^2 x - y(x^2 + y^2)/4           Z2Y                     *
*   3     2            z(x^2 - y^2)                ZX2Y2                    *
*   3    -2                zxy                      ZXY                     *
*   3     3            x^3 - 3x^2 y                  X3                     *
*   3    -3            y^3 - 3y^2 x                  Y3                     *
*                                                                           *
* Usage:  shimfield(l,m,x,y,z) returns a double precision value of the shim *
* field with indices (l,m) at cartesian coordinates x,y, and z.             *
*                                                                           * 
****************************************************************************/



double shim_field(int l, int m, float x, float y, float z)
{

  double B;

  if(l==1){
    if(m==0){
      B = z;
    }
    else if(m==1){
      B = y;
    }
    else if(m==-1){
      B = x;
    }
  }
  else if(l==2){
    if(m==0){
      B = z*z - (x*x + y*y)/2.0;
    }
    else if(m==1){
      B = z*y;
    }
    else if(m==-1){
      B = z*x;
    }
    else if(m==2){
      B = x*x - y*y;
    }
    else if(m==-2){
      B = x*y;
    }
  }
  else if(l==3){
    if(m==0){
      B = z*z*z - 3.0*z*(x*x + y*y)/2.0;
    }
    else if(m==1){
      B = z*z*x - x*(x*x + y*y)/4.0;
    }
    else if(m==-1){
      B = z*z*x - y*(x*x + y*y)/4.0;
    }
    else if(m==2){
      B = z*(x*x - y*y);
    }
    else if(m==-2){
      B = z*x*y;
    }
    else if(m==3){
      B = x*x*x - 3.0*x*x*y;
    }
    else if(m==-3){
      B = y*y*y - 3.0*y*y*x;
    }
  }

  return B;
}



