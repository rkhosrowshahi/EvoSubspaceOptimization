#ifndef _F21_H
#define _F21_H

#include "Benchmarks.h"

class F21:public Benchmarks{
public:
	F21();
	double compute(double* x) ;
	/* double compute(vector<double> x) ; */
	~F21();
};
#endif
