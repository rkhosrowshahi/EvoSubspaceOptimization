#ifndef _F20_H
#define _F20_H

#include "Benchmarks.h"

class F20:public Benchmarks{
 protected:
  void generateInterArray ( );
 public:
  F20();
  double compute(double* x) ;
  ~F20();
};

#endif
