/*************************************************************************

  snaphu input/output source file

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


/* function: SetDefaults()
 * -----------------------
 * Sets all parameters to their initial default values.
 */
//void SetDefaults(infileT *infiles, outfileT *outfiles, paramT *params){
void SetDefaults(paramT *params){

  /* input files */
/*  StrNCopy(infiles->weightfile,DEF_WEIGHTFILE,MAXSTRLEN);
  StrNCopy(infiles->corrfile,DEF_CORRFILE,MAXSTRLEN);
  StrNCopy(infiles->ampfile,DEF_AMPFILE,MAXSTRLEN);
  StrNCopy(infiles->ampfile2,DEF_AMPFILE2,MAXSTRLEN);
  StrNCopy(infiles->estfile,DEF_ESTFILE,MAXSTRLEN);  
  StrNCopy(infiles->magfile,DEF_MAGFILE,MAXSTRLEN);
  StrNCopy(infiles->costinfile,DEF_COSTINFILE,MAXSTRLEN);
*/
  /* output and dump files */
/*  StrNCopy(outfiles->initfile,DEF_INITFILE,MAXSTRLEN);
  StrNCopy(outfiles->flowfile,DEF_FLOWFILE,MAXSTRLEN);
  StrNCopy(outfiles->eifile,DEF_EIFILE,MAXSTRLEN);
  StrNCopy(outfiles->rowcostfile,DEF_ROWCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->colcostfile,DEF_COLCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstrowcostfile,DEF_MSTROWCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstcolcostfile,DEF_MSTCOLCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstcostsfile,DEF_MSTCOSTSFILE,MAXSTRLEN);
  StrNCopy(outfiles->corrdumpfile,DEF_CORRDUMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->rawcorrdumpfile,DEF_RAWCORRDUMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->costoutfile,DEF_COSTOUTFILE,MAXSTRLEN);
  StrNCopy(outfiles->conncompfile,DEF_CONNCOMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->outfile,DEF_OUTFILE,MAXSTRLEN);  
  StrNCopy(outfiles->logfile,DEF_LOGFILE,MAXSTRLEN);
*/
  /* file formats */
/*  infiles->infileformat=DEF_INFILEFORMAT;
  infiles->unwrappedinfileformat=DEF_UNWRAPPEDINFILEFORMAT;
  infiles->magfileformat=DEF_MAGFILEFORMAT;
  infiles->corrfileformat=DEF_CORRFILEFORMAT;
  infiles->estfileformat=DEF_ESTFILEFORMAT;
  infiles->ampfileformat=DEF_AMPFILEFORMAT;
  outfiles->outfileformat=DEF_OUTFILEFORMAT;
*/
  /* options and such */
  params->unwrapped=DEF_UNWRAPPED;
  params->regrowconncomps=DEF_REGROWCONNCOMPS;
  params->eval=DEF_EVAL;
  params->initonly=DEF_INITONLY;
  params->initmethod=DEF_INITMETHOD;
  params->costmode=DEF_COSTMODE;
  params->amplitude=DEF_AMPLITUDE;
  params->verbose=DEF_VERBOSE;

  /* SAR and geometry parameters */
  params->orbitradius=DEF_ORBITRADIUS;
  params->altitude=DEF_ALTITUDE;
  params->earthradius=DEF_EARTHRADIUS;
  params->bperp=DEF_BPERP; 
  params->transmitmode=DEF_TRANSMITMODE;
  params->baseline=DEF_BASELINE;
  params->baselineangle=DEF_BASELINEANGLE;
  params->nlooksrange=DEF_NLOOKSRANGE;
  params->nlooksaz=DEF_NLOOKSAZ;
  params->nlooksother=DEF_NLOOKSOTHER;
  params->ncorrlooks=DEF_NCORRLOOKS;           
  params->ncorrlooksrange=DEF_NCORRLOOKSRANGE;
  params->ncorrlooksaz=DEF_NCORRLOOKSAZ;
  params->nearrange=DEF_NEARRANGE;         
  params->dr=DEF_DR;               
  params->da=DEF_DA;               
  params->rangeres=DEF_RANGERES;         
  params->azres=DEF_AZRES;            
  params->lambda=DEF_LAMBDA;           

  /* scattering model parameters */
  params->kds=DEF_KDS;
  params->specularexp=DEF_SPECULAREXP;
  params->dzrcritfactor=DEF_DZRCRITFACTOR;
  params->shadow=DEF_SHADOW;
  params->dzeimin=DEF_DZEIMIN;
  params->laywidth=DEF_LAYWIDTH;
  params->layminei=DEF_LAYMINEI;
  params->sloperatiofactor=DEF_SLOPERATIOFACTOR;
  params->sigsqei=DEF_SIGSQEI;

  /* decorrelation model parameters */
  params->drho=DEF_DRHO;
  params->rhosconst1=DEF_RHOSCONST1;
  params->rhosconst2=DEF_RHOSCONST2;
  params->cstd1=DEF_CSTD1;
  params->cstd2=DEF_CSTD2;
  params->cstd3=DEF_CSTD3;
  params->defaultcorr=DEF_DEFAULTCORR;
  params->rhominfactor=DEF_RHOMINFACTOR;

  /* pdf model parameters */
  params->dzlaypeak=DEF_DZLAYPEAK;
  params->azdzfactor=DEF_AZDZFACTOR;
  params->dzeifactor=DEF_DZEIFACTOR;
  params->dzeiweight=DEF_DZEIWEIGHT;
  params->dzlayfactor=DEF_DZLAYFACTOR;
  params->layconst=DEF_LAYCONST;
  params->layfalloffconst=DEF_LAYFALLOFFCONST;
  params->sigsqshortmin=DEF_SIGSQSHORTMIN;
  params->sigsqlayfactor=DEF_SIGSQLAYFACTOR;
  
  /* deformation mode parameters */
  params->defoazdzfactor=DEF_DEFOAZDZFACTOR;
  params->defothreshfactor=DEF_DEFOTHRESHFACTOR;
  params->defomax=DEF_DEFOMAX;
  params->sigsqcorr=DEF_SIGSQCORR;
  params->defolayconst=DEF_DEFOLAYCONST;

  /* algorithm parameters */
  params->flipphasesign=DEF_FLIPPHASESIGN;
  params->initmaxflow=DEF_INITMAXFLOW;
  params->arcmaxflowconst=DEF_ARCMAXFLOWCONST;
  params->maxflow=DEF_MAXFLOW;
  params->krowei=DEF_KROWEI;
  params->kcolei=DEF_KCOLEI;   
  params->kperpdpsi=DEF_KPERPDPSI;
  params->kpardpsi=DEF_KPARDPSI;
  params->threshold=DEF_THRESHOLD;  
  params->initdzr=DEF_INITDZR;    
  params->initdzstep=DEF_INITDZSTEP;    
  params->maxcost=DEF_MAXCOST;
  params->costscale=DEF_COSTSCALE;      
  params->costscaleambight=DEF_COSTSCALEAMBIGHT;      
  params->dnomincangle=DEF_DNOMINCANGLE;
  params->srcrow=DEF_SRCROW;
  params->srccol=DEF_SRCCOL;
  params->p=DEF_P;
  params->nshortcycle=DEF_NSHORTCYCLE;
  params->maxnewnodeconst=DEF_MAXNEWNODECONST;
  params->maxcyclefraction=DEF_MAXCYCLEFRACTION;
  params->sourcemode=DEF_SOURCEMODE;
  params->maxnflowcycles=DEF_MAXNFLOWCYCLES;
  params->dumpall=DEF_DUMPALL;
  params->cs2scalefactor=DEF_CS2SCALEFACTOR;

  /* tile parameters */
  params->ntilerow=DEF_NTILEROW;
  params->ntilecol=DEF_NTILECOL;
  params->rowovrlp=DEF_ROWOVRLP;
  params->colovrlp=DEF_COLOVRLP;
  params->piecefirstrow=DEF_PIECEFIRSTROW;
  params->piecefirstcol=DEF_PIECEFIRSTCOL;
  params->piecenrow=DEF_PIECENROW;
  params->piecencol=DEF_PIECENCOL;
  params->tilecostthresh=DEF_TILECOSTTHRESH;
  params->minregionsize=DEF_MINREGIONSIZE;
  params->nthreads=DEF_NTHREADS;
  params->scndryarcflowmax=DEF_SCNDRYARCFLOWMAX;
  params->assembleonly=DEF_ASSEMBLEONLY;
  params->rmtmptile=DEF_RMTMPTILE;
  params->tileedgeweight=DEF_TILEEDGEWEIGHT;

  /* connected component parameters */
  params->minconncompfrac=DEF_MINCONNCOMPFRAC;
  params->conncompthresh=DEF_CONNCOMPTHRESH;
  params->maxncomps=DEF_MAXNCOMPS;

}

/* function: CheckParams()
 * -----------------------
 * Checks all parameters to make sure they are valid.  This is just a boring
 * function with lots of checks in it.
 */
void CheckParams(long linelen, long nlines, paramT *params){

  long ni, nj, n;
  FILE *fp;
  
  if(params->maxnflowcycles==USEMAXCYCLEFRACTION){
    params->maxnflowcycles=LRound(params->maxcyclefraction
				   *nlines/(double )params->ntilerow
				   *linelen/(double )params->ntilecol);
  }

  params->piecefirstrow--;                   /* index from 0 instead of 1 */
  params->piecefirstcol--;                   /* index from 0 instead of 1 */
  if(!params->piecenrow){
    params->piecenrow=nlines;
  }
  if(!params->piecencol){
    params->piecencol=linelen;
  }
  if(params->piecefirstrow<0 || params->piecefirstcol<0 
     || params->piecenrow<1 || params->piecencol<1
     || params->piecefirstrow+params->piecenrow>nlines
     || params->piecefirstcol+params->piecencol>linelen){
    //fprintf(sp0,"illegal values for piece of interferogram to unwrap\n");
    exit(ABNORMAL_EXIT);
  }

  /* set global pointers to functions for calculating and evaluating costs */
  if(params->p<0){
    if(params->costmode==TOPO){
      CalcCost=CalcCostTopo;
      EvalCost=EvalCostTopo;
    }else if(params->costmode==DEFO){
      CalcCost=CalcCostDefo;
      EvalCost=EvalCostDefo;
    }else if(params->costmode==SMOOTH){
      CalcCost=CalcCostSmooth;
      EvalCost=EvalCostSmooth;
    }
  }else{
    if(params->p==0){
      CalcCost=CalcCostL0;
      EvalCost=EvalCostL0;
    }else if(params->p==1){
      CalcCost=CalcCostL1;
      EvalCost=EvalCostL1;
    }else if(params->p==2){
      CalcCost=CalcCostL2;
      EvalCost=EvalCostL2;
    }else{
      CalcCost=CalcCostLP;
      EvalCost=EvalCostLP;
    }
  }
}

