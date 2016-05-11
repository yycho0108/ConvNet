#include "SoftMaxLayer.h"


__global__ void softMax_half(double* I, double* O, double max_in){
	int i = threadIdx.x;
	int n = blockDim.x;
	O[i] = exp(I[i]-max_in);
}

void softMax(Matrix& I, Matrix& O){
	double max_in = I.max();
	softMax_half<<<1,I.size().wh>>>(I.d_data(), O.d_data(), max_in);
	O /= O.sum();
}

SoftMaxLayer::SoftMaxLayer(){

}
SoftMaxLayer::~SoftMaxLayer(){

}
void SoftMaxLayer::setup(Size& s, int& d){
	this->s = s;
	this->d = d;
	O.push_back(Matrix(s));
	G.push_back(Matrix(s));
}

std::vector<Matrix>& SoftMaxLayer::FF(std::vector<Matrix>& _I){
	I.swap(_I);
	for(int i=0;i<d;++i){
		softMax(I[i],O[i]);
	}
	return O;
}
std::vector<Matrix>& SoftMaxLayer::BP(std::vector<Matrix>& _G){
	throw "SoftMAX : BP Not Implemented";
	G.swap(_G);
	return G;
}

void SoftMaxLayer::update(){

}
