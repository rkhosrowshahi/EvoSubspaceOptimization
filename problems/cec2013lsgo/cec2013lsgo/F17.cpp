#include "F17.h"

F17::F17():Benchmarks(){
  Ovector = NULL;
  minX = -5;
  maxX = 5;
  ID = 17;
  dimension = 10000;
  anotherz = new double[dimension];
}

F17::~F17(){
  delete[] Ovector;
  delete[] anotherz;
}

double F17::compute(double* x){
  int    i;
  double result;

  if(Ovector == NULL) {
    // Ovector = createShiftVector(dimension,minX,maxX);
    Ovector = readOvector();
  }

  for(i = 0; i < dimension; i++) {
    anotherz[i] = x[i] - Ovector[i];
  }
  
  // // T_{osz}
  // transform_osz(anotherz, dimension);
  
  // // T_{asy}^{0.2}
  // transform_asy(anotherz, 0.2);

  // // lambda
  // Lambda(anotherz, 10);

  result = rastrigin(anotherz,dimension);
  update(result);
  return(result);
}
