/*
 * ConvolutionLayer.cpp
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#include "ConvolutionLayer.h"

void convolve(Matrix& I, Matrix& K, Matrix& O) {
	int n = I.size().n;
	int m = I.size().m;
	int r = K.size().n / 2;

	convolve_d(I.d_data(), K.d_data(), O.d_data(), n, m, r);
}

__global__ double deconvolve(double* I, double* G, double* O, int ix, int iy, int gx, int gy, int w, int h){


}


void deconvolve(Matrix& I, Matrix& _G, Matrix& dW){
	auto s = dW.size().n;
	auto r = dW.size().n / 2; //assume square kernel, odd-size


	for (int y = 0; y < s; ++y) {
		for (int x = 0; x < s; ++x) {

			auto width = iw - abs(x-r);
			auto height = ih - abs(y-r);

			deconvolve <<<1,1>>>(I.d_data(), _G.d_data(),
					max(0,r-x), max(0,r-y),
					max(0,x-r), max(0,y-r),
					width,height);

			dW(y, x) += cv::sum(I_dw.mul(G_dw))[0];
		}
	}
}

ConvolutionLayer::ConvolutionLayer(int d_out) :
		d_out(d_out) {
	m = momentum; //learning momentum, defined in Params.cpp

	// TODO Auto-generated constructor stub
}

void ConvolutionLayer::setup(Size& _s, int& _d) {
	//_d = depth of input

	s = _s;
	d_in = _d;

	for (int i = 0; i < d_in; ++i) {
		I.push_back(Matrix::empty(s));
	}

	for (int o = 0; o < d_out; ++o) {
		W.push_back(Matrix::rand(5, 5)); //5,5 = kernel size
		//Bias depends on output matrix size,
		//Which is equivalent to the input matrix size (in case of this convolution)
		dW.push_back(Matrix::zeros(5, 5));
		dW_p.push_back(Matrix::zeros(5, 5)); //previous dW

		G.push_back(Matrix::zeros(s));
		B.push_back(Matrix::zeros(s));
		dB.push_back(Matrix::zeros(s));
		dB_p.push_back(Matrix::zeros(s)); //previous dB
	}

	for (int o = 0; o < d_out; ++o) {
		connection[o] = new bool[d_i];
		for (int i = 0; i < d_in; ++i) {
			//connection[o][i] = true;
			connection[o][i] = ((o % 3) == (i % 3));
			//partial connection
		}
	}
	_s = s; //same size for current convolution function.
	_d = d_out;
}

ConvolutionLayer::~ConvolutionLayer() {
	// TODO Auto-generated destructor stub
}

std::vector<Matrix>& ConvolutionLayer::FF(std::vector<Matrix>& _I) {

	for (int i = 0; i < d_i; ++i) {
		_I[i].copyTo(I[i]);
		activate(I[i], O[i], f);
	}

	G = std::vector<Matrix>(I.size()); //Gradient

	Matrix tmp = Matrix::empty(W[0].size());

	for (int o = 0; o < d_out; ++o) {
		O[o].zero(); //set to zero
		for (int i = 0; i < d_in; ++i) {
			if (connection[o][i]) {
				convolve(I[i], W[o], tmp);
				O[o] += tmp;
			}
		}
		O[o] += B[o]; //add bias
	}

	return O;
}

std::vector<Matrix>& ConvolutionLayer::	BP(std::vector<Matrix>& _G) {
	auto iw = s.width;
	auto ih = s.height;

	auto fw = W[0].size().width; //kernel size
	auto fh = W[0].size().height;

	auto fwr = fw / 2; //kernel size
	auto fhr = fh / 2;

	for (int i = 0; i < d_i; ++i) {
		G[i].zero(); //reset to 0
	}

	for (int o = 0; o < d_o; ++o) { //for each output channel(depth):
		dW[o].zero();

		for (int i = 0; i < d_i; ++i) { //for each input channel
			if (connection[o][i]) { //if the channels are related..

				Mat tmp;
				correlate(_G[o],W[o],tmp); //TODO : implement to correlation
				G[i] += tmp;
				deconvolve(I,_G,tmp);
			}
		}
		dW[o] = (m * dW_p[o])
				+ (ETA * g[o] % dW[o])
				- W[o] * DECAY;

		dB[o] = (m * db_p[o])
				+ (ETA * _G[o]); //bias = gradient
				- B[o] * DECAY;

		dW_p[o] = dW[o];
		dB_p[o] = dB[o];
	}
	return G;
}
