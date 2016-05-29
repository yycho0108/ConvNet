/*
 * RandManager.h
 *
 *  Created on: May 11, 2016
 *      Author: jamiecho
 */

#ifndef RANDMANAGER_H_
#define RANDMANAGER_H_

#include <curand.h>
#include <curand_kernel.h>

class RandManager {
private:
	curandGenerator_t rgen;
public:
	RandManager();
	~RandManager();
	void rand(double* a, int n);
	void randu(double* a, int n);
	void randn(double* a, int n, double mean=0.0, double stddev=1.0);

};


#endif /* RANDMANAGER_H_ */

