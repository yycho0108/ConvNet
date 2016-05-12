#include "FlattenLayer.h"

FlattenLayer::FlattenLayer() {
	d_in=0;
}

FlattenLayer::~FlattenLayer() {

}

void FlattenLayer::setup(Size& s, int& d) {
	d_in = d;
	s_in = s;
	s_out = Size(1, d * s.wh);

	O.push_back(Matrix(s_out));
	for (int i = 0; i < d_in; ++i) {
		I.push_back(Matrix(s_in)); //preallocating
		G.push_back(Matrix(s_in));
	}

	s = s_out;
	d = 1;
}

std::vector<Matrix>& FlattenLayer::FF(std::vector<Matrix>& _I) {
	double* o_ptr = O[0].d_data();
	auto sz = s_in.wh * sizeof(double);

	for (int i = 0; i < d_in; ++i) {
		_I[i].copyTo(I[i]);

		cudaMemcpy(o_ptr + i*s_in.wh, I[i].d_data(), sz,
				cudaMemcpyDeviceToDevice);
	}
	return O;
}

std::vector<Matrix>& FlattenLayer::BP(std::vector<Matrix>& _G) {
	double* g_ptr = _G[0].d_data();
	auto sz = s_in.wh * sizeof(double);

	for (int i = 0; i < d_in; ++i) {
		cudaMemcpy(G[i].d_data(), g_ptr + i*s_in.wh, sz,
				cudaMemcpyDeviceToDevice);
	}

	return G;
}

void FlattenLayer::update(){

}
