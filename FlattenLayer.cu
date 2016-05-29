#include "FlattenLayer.h"

FlattenLayer::FlattenLayer() {
	d_in=0;
}

FlattenLayer::~FlattenLayer() {
	delete[] streams;
}

void FlattenLayer::setup(Size& s, int& d) {
	d_in = d;
	s_in = s;
	s_out = Size(1, d * s.wh);

	O.push_back(Matrix(s_out));

	streams = new cudaStream_t[d_in];
	for (int i = 0; i < d_in; ++i) {
		G.push_back(Matrix(s_in));
		cudaStreamCreate(&streams[i]);
	}

	s = s_out;
	d = 1;
}

std::vector<Matrix>& FlattenLayer::FF(std::vector<Matrix>& _I) {
	double* o_ptr = O[0].d_data();
	auto sz = s_in.wh * sizeof(double);

	for (int i = 0; i < d_in; ++i) {
		//cudaMemcpy(o_ptr + i*s_in.wh, _I[i].d_data(), sz, cudaMemcpyDeviceToDevice);
		cudaMemcpyAsync(o_ptr + i*s_in.wh, _I[i].d_data(), sz,
				cudaMemcpyDeviceToDevice,streams[i]);
	}
	for(int i=0;i<d_in;++i){
		cudaStreamSynchronize(streams[i]);
	}
	return O;
}

std::vector<Matrix>& FlattenLayer::BP(std::vector<Matrix>& _G) {
	double* g_ptr = _G[0].d_data();
	auto sz = s_in.wh * sizeof(double);

	for (int i = 0; i < d_in; ++i) {
		//cudaMemcpy(G[i].d_data(), g_ptr + i*s_in.wh, sz,
		//				cudaMemcpyDeviceToDevice);
		cudaMemcpyAsync(G[i].d_data(), g_ptr + i*s_in.wh, sz,
				cudaMemcpyDeviceToDevice,streams[i]);
	}

	for(int i=0;i<d_in;++i){
		cudaStreamSynchronize(streams[i]);
	}

	return G;
}

void FlattenLayer::update(){

}
