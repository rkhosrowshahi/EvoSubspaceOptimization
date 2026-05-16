#ifndef _F23_H
#define _F23_H

#include "Benchmarks.h"


class F23:public Benchmarks{
protected:

public:
	F23();
	double compute(double* x) ;
	/* double compute(vector<double> x) ; */
	~F23();
};
#endif
