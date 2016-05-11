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
	int N;
	curandState* s;
public:
	RandManager(int N);
	~RandManager();
	void rand(double* a, int n);
};


#endif /* RANDMANAGER_H_ */

