#ifndef _F24_H
#define _F24_H

#include "Benchmarks.h"

class F24:public Benchmarks{
 protected:
 public:
  F24();
  double compute(double* x) ;
  ~F24();
};

#endif
