#ifndef _F25_H
#define _F25_H

#include "Benchmarks.h"

class F25:public Benchmarks{
 protected:
  void generateInterArray ( );
 public:
  F25();
  double compute(double* x) ;
  ~F25();
};

#endif
