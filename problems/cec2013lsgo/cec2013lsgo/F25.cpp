#include "F25.h"
#include <stdio.h>

F25::F25():Benchmarks(){
  Ovector = NULL;
  minX = -100;
  maxX = 100;
  ID = 25;
  dimension = 100000;
  anotherz = new double[dimension];
}

F25::~F25(){
  delete[] Ovector;
  delete[] anotherz;
}

double F25::compute(double*x){
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

  result=schwefel(anotherz, dimension);

  update(result);
  return(result);
}
