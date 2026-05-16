#ifndef _F19_H
#define _F19_H

#include "Benchmarks.h"

class F19:public Benchmarks{
 protected:
 public:
  F19();
  double compute(double* x) ;
  ~F19();
};

#endif
