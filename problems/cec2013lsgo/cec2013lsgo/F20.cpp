#include "F20.h"
#include <stdio.h>

F20::F20():Benchmarks(){
  Ovector = NULL;
  minX = -100;
  maxX = 100;
  ID = 20;
  dimension = 10000;
  anotherz = new double[dimension];
}

F20::~F20(){
  delete[] Ovector;
  delete[] anotherz;
}

double F20::compute(double*x){
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
