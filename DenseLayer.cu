#include "DenseLayer.h"

DenseLayer::DenseLayer(int s_o) :
		s_o(s_o) {
}

DenseLayer::~DenseLayer() {

}

void DenseLayer::setup(Size& s, int& d) {
	//assert(s.w == 1);
	//assert(d == 1);
	s_i = s.h; //column vector input

	//CAUTION:: do not use s directly for the size!
	//it is not the dimension of the kernel.

	W = Matrix::rand(s_i, s_o); //width = s_i, height = s_o
	B = Matrix::zeros(1, s_o);

	dW = Matrix::zeros(s_i, s_o);
	dB = Matrix::zeros(1, s_o);

	dW_p = Matrix::zeros(s_i, s_o);
	dB_p = Matrix::zeros(1, s_o);

	dW_t = Matrix::zeros(s_i, s_o);
	dB_t = Matrix::zeros(1, s_o);

	//placeholders

	I.push_back(Matrix());
	O.push_back(Matrix());
	G.push_back(Matrix());

	s = Size(1,s_o);
	d = 1;
	cudaStreamCreate(&stream);
}

std::vector<Matrix>& DenseLayer::FF(std::vector<Matrix>& _I) {
	//pI = &_I;
	_I[0].copyTo(I[0]);

	//namedPrint(I[0]);
	O[0] = W * I[0] + B;
	//namedPrint(O[0]);
	return O; //no activation! add it separately.
}

std::vector<Matrix>& DenseLayer::BP(std::vector<Matrix>& _G) {
	//std::vector<Matrix>& I = *pI;

	//TODO : implement fancy optimizations
	Matrix Wt = Matrix::transpose(W);
	G[0] = Wt * _G[0];
	/*dW = (dW_p * MOMENTUM) // momentum * previous dW
			+ (_G[0] * Matrix::transpose(I[0]) * ETA) // learning rate * weight error
			- (W * DECAY); //decay * weight

	dB = (dB_p * MOMENTUM)
		+ (_G[0] * ETA)
		- (B * DECAY)*/
	dW = _G[0] * Matrix::transpose(I[0]);

	dW_t += dW;
	dB_t += _G[0];

	//W += dW * ETA;
	//B += _G[0] * ETA; //individual updates

	//dW.copyTo(dW_p);
	//dB.copyTo(dB_p);

	//dW_p = dW;
	//dB_p = dB;

	return G;
}

void DenseLayer::update(){
	//dW_t /= 128.0;
	//dB_t /= 128.0;

	W += (dW_p * MOMENTUM) + \
		 (dW_t * ETA) - \
		 (W * DECAY);
	B += (dB_p * MOMENTUM) + \
		 (dW_t * ETA);

	dW_t.copyTo(dW_p);
	dB_t.copyTo(dB_p);

	dW_t.zero();
	dB_t.zero();
}
