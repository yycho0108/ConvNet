#include "SoftMaxLayer.h"

__global__ void softMax_half(double* I, double* O, double max_in) {
	int i = threadIdx.x;
	O[i] = exp(I[i] - max_in);
}

void softMax(Matrix& I, Matrix& O) {
	double max_in = I.max();
	//namedPrint(I);
	// currently implemented in host, due to strange bug.
	/*I.copyTo(O);
	O.sync();

	double accum;
	double* o_ptr = O.data();
	for (int i = 0; i < O.size().wh; ++i) {
		o_ptr[i] = exp(o_ptr[i]);
		accum += o_ptr[i];
	}
	for (int i = 0; i < O.size().wh; ++i) {
		o_ptr[i] /= accum;
	}

	O.sync_r();*/

	//device code:
	softMax_half<<<1,I.size().wh>>>(I.d_data(), O.d_data(), max_in);
	O.set_sync(false);
	O /= O.sum();
	//namedPrint(O);
}

SoftMaxLayer::SoftMaxLayer() {

}
SoftMaxLayer::~SoftMaxLayer() {

}
void SoftMaxLayer::setup(Size& s, int& d) {
	this->s = s;
	this->d = d;
	I.push_back(Matrix(s));
	O.push_back(Matrix(s));
	G.push_back(Matrix(s));
}

std::vector<Matrix>& SoftMaxLayer::FF(std::vector<Matrix>& _I) {
	for (int i = 0; i < d; ++i) {
		_I[i].copyTo(I[i]);
		softMax(I[i], O[i]);
	}
	return O;
}
std::vector<Matrix>& SoftMaxLayer::BP(std::vector<Matrix>& _G) {
	throw "SoftMAX : BP Not Implemented";
	for (int i = 0; i < d; ++i) {
		_G[i].copyTo(G[i]);
	}
	return G;
}

void SoftMaxLayer::update() {

}
