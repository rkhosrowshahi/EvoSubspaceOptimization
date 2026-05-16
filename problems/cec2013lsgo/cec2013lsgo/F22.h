#ifndef _F22_H
#define _F22_H

#include "Benchmarks.h"


class F22:public Benchmarks{
protected:

public:
	F22();
	double compute(double* x) ;
	/* double compute(vector<double> x) ; */
	~F22();
};
#endif
