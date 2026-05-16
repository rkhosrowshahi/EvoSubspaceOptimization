#include "F24.h"
#include <stdio.h>

F24::F24():Benchmarks(){
  Ovector = NULL;
  minX = -100;
  maxX = 100;
  ID = 24;
  dimension = 100000;
  anotherz = new double[dimension];
}

F24::~F24(){
  delete[] Ovector;
  delete[] anotherz;
}

double F24::compute(double*x){
  int i;
  double result=0.0;

  if(Ovector==NULL)
    {
      Ovector = readOvector();
    }

  for(i=0;i<dimension;i++)
    {
      anotherz[i]=x[i]-Ovector[i];
    }

  result=rosenbrock(anotherz, dimension);
  update(result);
  return(result);
}

