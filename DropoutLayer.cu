#include "DropoutLayer.h"

bool DropoutLayer::enabled = true;

DropoutLayer::DropoutLayer(double p):p(p){

}

DropoutLayer::~DropoutLayer(){

}
void DropoutLayer::setup(Size& _s, int& _d) {
	s = _s;
	d = _d;

	for (int i = 0; i < d; ++i) {
		I.push_back(Matrix(s));
		G.push_back(Matrix(s));
		O.push_back(Matrix(s));
		Mask.push_back(Matrix(s));
	}

}

std::vector<Matrix>& DropoutLayer::FF(std::vector<Matrix>& _I) {
	if(enabled){
		for (int i = 0; i < d; ++i) {
				_I[i].copyTo(I[i]);
				Mask[i].randu(0.0,1.0);
				Mask[i] = (Mask[i] < p); //binary threshold
				O[i] = I[i] % Mask[i];
			}
		return O;
	}else{
		return _I;
	}
}

std::vector<Matrix>& DropoutLayer::BP(std::vector<Matrix>& _G) {
	if(enabled){
		for (int i = 0; i < d; ++i) {
			G[i] = _G[i] % Mask[i];
			G[i] /= p;
		}
		return G;
	}else{
		return _G;
	}
}


void DropoutLayer::update(){

}

void DropoutLayer::enable(bool d){
	enabled = d;
}
