#include "EdgesCalculations.h"
#include "LinkedListFunctions.h"

void Intialize_Voxels(Voxel * Voxels_Pointer,float *Phase,float *Quality,int x, int y, int z)
{
  int i,j,k,frame,row,index;
  Voxel *temp = Voxels_Pointer;
	
  frame = x*y;
  row = x;
  for(k=0; k<z; k++)
    {
    for(j=0; j<y; j++)
      {
      for(i=0; i<x; i++)
        {
        index = k*frame + j*row + i;
        temp->x = i;
        temp->y = j;
        temp->z = k;
        temp->value = Phase[index];
        temp->Reliability = Quality[index];
        temp->freq = 1;
        temp->increment = 0;
        temp->head = temp;
        temp->next = NULL;
        temp->tail = temp;
        temp++;
        }
      }
    }
}

void HorizantalEdges(Voxel *voxel_pointer, Edge **EdgeHead,Edge **EdgeTail, int x, int y, int z,int *EdgeIndex)
{
  int i,j,k,Frame,Row;
  float Rel=0;
  Frame = x*y;
  Row = x;

  Voxel *voxel_temp = voxel_pointer+Frame+Row+1;
  Edge *temp;
	
  for ( k =1; k<z-1; k++)
    {
    for(j=1; j<y-1; j++)
      {
      for(i=1; i<x-2; i++)
        {
        Rel = voxel_temp->Reliability + (voxel_temp+1)->Reliability;
        temp = CreateNewNode(voxel_temp,(voxel_temp+1),Rel,(*EdgeIndex));
        Add2End(EdgeHead,EdgeTail,temp);
        (*EdgeIndex)++;
        voxel_temp++;
        }
      voxel_temp +=3;
      }
    voxel_temp = voxel_temp + 2*Row;
    }
}

void VerticalEdges(Voxel *voxel_pointer, Edge **EdgeHead,Edge **EdgeTail, int x, int y, int z,int *EdgeIndex)
{
  int i,j,k,Frame,Row;
  float Rel;
  Frame = x*y;
  Row = x;
  Voxel *voxel_temp = voxel_pointer+Frame+Row+1;
  Edge *temp;
	
  for (k=1; k<z-1; k++)
    {
    for(j=1; j<y-2; j++)
      {
      for(i=1; i<x-1; i++)
        {
        Rel = voxel_temp->Reliability + (voxel_temp+Row)->Reliability;
        temp = CreateNewNode(voxel_temp,(voxel_temp+Row),Rel,(*EdgeIndex));
        Add2End(EdgeHead,EdgeTail,temp);
        (*EdgeIndex)++;
        voxel_temp++;
        }
      voxel_temp +=2;
      }
    voxel_temp += 3*Row;
    }

}

void NormalEdges(Voxel *voxel_pointer, Edge **EdgeHead,Edge **EdgeTail, int x, int y, int z,int *EdgeIndex)
{
  int i,j,k,Frame,Row;
  float Rel;
  Frame = x*y;
  Row = x;
  Voxel *voxel_temp = voxel_pointer+Frame+Row+1;
  Edge *temp;
	
  for ( k =1; k<z-2; k++)
    {
    for(j=1; j<y-1; j++)
      {
      for(i=1; i<x-1; i++)
        {
        Rel = voxel_temp->Reliability + (voxel_temp+Frame)->Reliability;
        temp = CreateNewNode(voxel_temp,(voxel_temp+Frame),Rel,(*EdgeIndex));
        Add2End(EdgeHead,EdgeTail,temp);
        (*EdgeIndex)++;
        voxel_temp++;
        }
      voxel_temp +=2;
      }
    voxel_temp +=2*Row;
    }
}
