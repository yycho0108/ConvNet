/*
 * Sampler.cpp
 *
 *  Created on: May 30, 2016
 *      Author: jamiecho
 */

#include "Sampler.h"
#include <algorithm>

Sampler::Sampler(rng_type::result_type seed) {
	rng.seed(seed);
}

std::vector<int> Sampler::operator()(int range, int nSamples) {
	std::vector<int> res;
	std::uniform_int_distribution < rng_type::result_type > udist(0, range);

	for (int i = 0; i < nSamples; ++i) {
		int r;
		do {
			r = udist(rng);
		} while (std::find(res.begin(), res.end(), r) != res.end());
		res.push_back(r);
	}

	return res;
}
