/*************************************************************************

  snaphu main source file
  Written by Curtis W. Chen
  Copyright 2002 Board of Trustees, Leland Stanford Jr. University
  Please see the supporting documentation for terms of use.
  No warranty.

*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "snaphu.h"
#include "snaphu_unwrap.h"


/* global (external) variable definitions */

/* flags used for signal handling */
char dumpresults_global;
char requestedstop_global;

/* ouput stream pointers */
/* sp0=error messages, sp1=status output, sp2=verbose, sp3=verbose counter */
FILE *sp0, *sp1, *sp2, *sp3;

/* node pointer for marking arc not on tree in apex array */
/* this should be treated as a constant */
nodeT NONTREEARC[1];

/* pointers to functions which calculate arc costs */
void (*CalcCost)();
long (*EvalCost)();

/* pointers to functions for tailoring network solver to specific topologies */
nodeT *(*NeighborNode)();
void (*GetArc)();

void doUnwrap(float *wr_phase, float *uw_phase, long nrow, long ncol) {
  paramT params[1];
  tileparamT tileparams[1];
  float **phase2D, **uwphase2D, *dp;
  int l,m;
  
  SetDefaults(params);
  /*params->p=2;
  params->costmode=NOSTATCOSTS;
  */
  params->costmode=SMOOTH;

  CheckParams(ncol, nrow, params);
  tileparams->firstrow=params->piecefirstrow;
  tileparams->firstcol=params->piecefirstcol;
  tileparams->nrow=nrow;
  tileparams->ncol=ncol;
  
  phase2D = (float **) Get2DMem(nrow, ncol, sizeof(float *), sizeof(float));
  dp = wr_phase;
  for(l=0; l<nrow; l++) {
    memmove(phase2D[l], dp, ncol*sizeof(float));
    dp += ncol;
  }
  WrapPhase(phase2D, nrow, ncol);
  uwphase2D = UnwrapTile(phase2D, params, tileparams, nrow);
  dp = uw_phase;
  for(l=0; l<nrow; l++) {
    memmove(dp, uwphase2D[l], ncol*sizeof(float));
    dp += ncol;
  }
  Free2DArray((void **)phase2D, nrow);
  Free2DArray((void **)uwphase2D, nrow);
}

/* function: UnwrapTile()
 * ----------------------
 * This is the main phase unwrapping function for a single tile.
 */
float **UnwrapTile(float **wrappedphase, paramT *params, tileparamT *tileparams, long nlines){

  /* variable declarations */
  long nrow, ncol, nnoderow, narcrow, n, ngroundarcs, iincrcostfile;
  long nflow, ncycle, mostflow, nflowdone;
  long candidatelistsize, candidatebagsize;
  short *nnodesperrow, *narcsperrow;
  short **flows, **mstcosts;
  float **unwrappedphase, **mag, **unwrappedest;
  incrcostT **incrcosts;
  void **costs;
  totalcostT totalcost, oldtotalcost;
  nodeT *source, ***apexes;
  nodeT **nodes, ground[1];
  candidateT *candidatebag, *candidatelist;
  signed char **iscandidate;
  signed char notfirstloop;
  bucketT *bkts;


  /* get size of tile */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;

  /* read the coarse unwrapped estimate, if provided */
  unwrappedest=NULL;
  mag = (float **)Get2DMem(nrow, ncol, sizeof(float *), sizeof(float));
  int m,l;
  for(m=0; m<nrow; m++)
    for(l=0; l<ncol; l++)
      mag[m][l] = 1.0;
  
  /* build the cost arrays */  
  BuildCostArrays(&costs,&mstcosts,mag,wrappedphase,unwrappedest,
		  nlines,nrow,ncol,params,tileparams);
  /* set network function pointers for grid network */
  NeighborNode=NeighborNodeGrid;
  GetArc=GetArcGrid;

  /* initialize the flows (find simple unwrapping to get a feasible flow) */
  unwrappedphase=NULL;
  nodes=NULL;
  if(!params->unwrapped){

    /* see which initialization method to use */
    if(params->initmethod==MSTINIT){
      /* use minimum spanning tree (MST) algorithm */
      MSTInitFlows(wrappedphase,&flows,mstcosts,nrow,ncol,
		   &nodes,ground,params->initmaxflow);    
    }else{
      //fprintf(sp0,"Illegal initialization method\nAbort\n");
      exit(ABNORMAL_EXIT);
    }

    /* integrate the phase and write out if necessary */
  }

  /* initialize network variables */
  InitNetwork(flows,&ngroundarcs,&ncycle,&nflowdone,&mostflow,&nflow,
	      &candidatebagsize,&candidatebag,&candidatelistsize,
	      &candidatelist,&iscandidate,&apexes,&bkts,&iincrcostfile,
	      &incrcosts,&nodes,ground,&nnoderow,&nnodesperrow,&narcrow,
	      &narcsperrow,nrow,ncol,&notfirstloop,&totalcost,params);


  /* if we have a single tile, trap signals for dumping results */
  if(params->ntilerow==1 && params->ntilecol==1){
    signal(SIGINT,SetDump);
    signal(SIGHUP,SetDump);
  }

  /* main loop: loop over flow increments and sources */
  //fprintf(sp1,"Running nonlinear network flow optimizer\n");
  //fprintf(sp1,"Maximum flow on network: %ld\n",mostflow);
  //fprintf(sp2,"Number of nodes in network: %ld\n",(nrow-1)*(ncol-1)+1);
  while(TRUE){ 
 
    //fprintf(sp1,"Flow increment: %ld  (Total improvements: %ld)\n",
	  //  nflow,ncycle);
    
    /* set up the incremental (residual) cost arrays */
    SetupIncrFlowCosts(costs,incrcosts,flows,nflow,nrow,narcrow,narcsperrow,
		       params); 

    /* set the tree root (equivalent to source of shortest path problem) */
    source=SelectSource(nodes,ground,nflow,flows,ngroundarcs,
			nrow,ncol,params);

    /* run the solver, and increment nflowdone if no cycles are found */
    n=TreeSolve(nodes,NULL,ground,source,&candidatelist,&candidatebag,
		&candidatelistsize,&candidatebagsize,
		bkts,flows,costs,incrcosts,apexes,iscandidate,
		ngroundarcs,nflow,mag,wrappedphase,
		nnoderow,nnodesperrow,narcrow,narcsperrow,nrow,ncol,
		params);

    /* evaluate and save the total cost (skip if first loop through nflow) */
    if(notfirstloop){
      oldtotalcost=totalcost;
      totalcost=EvaluateTotalCost(costs,flows,nrow,ncol,NULL,params);
      if(totalcost>oldtotalcost || (n>0 && totalcost==oldtotalcost)){
	//fprintf(sp0,"Unexpected increase in total cost.  Breaking loop\n");
	break;
      }
    }

    /* consider this flow increment done if not too many neg cycles found */
    ncycle+=n;
    if(n<=params->maxnflowcycles){
      nflowdone++;
    }else{
      nflowdone=1;
    }

    /* find maximum flow on network */
    mostflow=Short2DRowColAbsMax(flows,nrow,ncol);

    /* break if we're done with all flow increments or problem is convex */
    if(nflowdone>=params->maxflow || nflowdone>=mostflow || params->p>=1.0){
      break;
    }

    /* update flow increment */
    nflow++;
    if(nflow>params->maxflow || nflow>mostflow){
      nflow=1;
      notfirstloop=TRUE;
    }
    //fprintf(sp2,"Maximum flow on network: %ld\n",mostflow);

  } /* end loop until no more neg cycles */


  /* if we have single tile, return signal handlers to default behavior */
  if(params->ntilerow==1 && params->ntilecol==1){
    signal(SIGINT,SIG_DFL);
    signal(SIGHUP,SIG_DFL);
  }

  /* free some memory */
  Free2DArray((void **)apexes,2*nrow-1);
  Free2DArray((void **)iscandidate,2*nrow-1);
  Free2DArray((void **)nodes,nrow-1);
  free(candidatebag);
  free(candidatelist);  
  free(bkts->bucketbase);


  /* free some more memory */
  Free2DArray((void **)incrcosts,2*nrow-1);

  /* evaluate and display the maximum flow and total cost */
  totalcost=EvaluateTotalCost(costs,flows,nrow,ncol,NULL,params);
  //fprintf(sp1,"Maximum flow on network: %ld\n",mostflow);
  //fprintf(sp1,"Total solution cost: %.9g\n",(double )totalcost);

  /* integrate the wrapped phase using the solution flow */
  //fprintf(sp1,"Integrating phase\n");
  unwrappedphase=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  IntegratePhase(wrappedphase,unwrappedphase,flows,nrow,ncol);
  /* reinsert the coarse estimate, if it was given */
  if(unwrappedest!=NULL){
    Add2DFloatArrays(unwrappedphase,unwrappedest,nrow,ncol);
  }

  /* flip the sign of the unwrapped phase array if it was flipped initially, */
  FlipPhaseArraySign(unwrappedphase,params,nrow,ncol);  
  /* free remaining memory and return */
  Free2DArray((void **)costs,2*nrow-1);
  Free2DArray((void **)mag,nrow);
  Free2DArray((void **)flows,2*nrow-1);
  free(nnodesperrow);
  free(narcsperrow);
  return unwrappedphase;

} /* end of UnwrapTile() */
