#include "F19.h"
#include <stdio.h>

F19::F19():Benchmarks(){
  Ovector = NULL;
  minX = -100;
  maxX = 100;
  ID = 19;
  dimension = 10000;
  anotherz = new double[dimension];
}

F19::~F19(){
  delete[] Ovector;
  delete[] anotherz;
}

double F19::compute(double*x){
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

