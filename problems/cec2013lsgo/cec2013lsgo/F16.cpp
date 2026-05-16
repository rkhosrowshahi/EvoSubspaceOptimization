#include "F16.h"

F16::F16():Benchmarks(){
  Ovector = NULL;
  minX = -100;
  maxX = 100;
  ID = 16;
  dimension = 10000;
  anotherz = new double[dimension];
}

F16::~F16(){
  delete[] Ovector;
  delete[] anotherz;
  // (Ovector);
}

double F16::compute(double* x) {
  double result;
  int    i;

  if(Ovector == NULL) {
    // Ovector = createShiftVector(dimension,minX,maxX);
    Ovector = readOvector();
  }

  for(i = dimension - 1; i >= 0; i--) {
    anotherz[i] = x[i] - Ovector[i];
  }

  // T_{OSZ}
  // transform_osz(anotherz,dimension);
  // result = elliptic(anotherz,dimension);
  result = elliptic(anotherz,dimension);
  update(result);
  return(result);
}

// double F16::compute(vector<double> x){
//   double result;
//   int    i;

//   if(Ovector == NULL) {
//     Ovector = createShiftVector(dimension,minX,maxX);
//   }

//   for(i = dimension - 1; i >= 0; i--) {
//     anotherz[i] = x[i] - Ovector[i];
//   }

//   result = elliptic(anotherz,dimension);
//   return(result);
// }
