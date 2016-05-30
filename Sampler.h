/*
 * Sampler.h
 *
 *  Created on: May 30, 2016
 *      Author: jamiecho
 */

#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <vector>
#include <random>

class Sampler{

private:
	using rng_type = std::mt19937;
	rng_type rng;
public:
	Sampler(const rng_type::result_type seed);

	std::vector<int> operator()(int range, int nSamples);

};


#endif /* SAMPLER_H_ */
