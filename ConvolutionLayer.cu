/*
 * ConvolutionLayer.cpp
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#include "ConvolutionLayer.h"

__device__ int d_abs(int x) {
	return x > 0 ? x : -x;
}

void convolve(Matrix& I, Matrix& K, Matrix& O) {
	//TODO : support different modes
	int w = I.size().w;
	int h = I.size().h;
	int r = K.size().w / 2;

	convolve_d(I.d_data(), K.d_data(), O.d_data(), w, h, r);
}

void correlate(Matrix& I, Matrix& K, Matrix& O) {
	//TODO : support different modes
	int w = I.size().w;
	int h = I.size().h;
	int r = K.size().w / 2;

	correlate_d(I.d_data(), K.d_data(), O.d_data(), w, h, r);
}

/*__device__ void submat_mul(double* a, double* b, double* o,
 int asrci, int asrcj, int aw,
 int bsrci, int bsrcj, int bw){

 auto i = threadIdx.y;
 auto j = threadIdx.x;
 auto w = blockDim.x;

 auto a_i = idx(asrci + i, asrcj + j, aw);
 auto b_i = idx(bsrci + i, bsrcj + j, bw);

 o[idx(i,j,w)] = a[a_i] * b[b_i];
 }*/

__global__ void deconvolve(double* I, double* _G, double* dW, int iw, int ih) {
	//TODO : current version is likely to be wrong... FIX!!

	//w_i,w_j spans to Kernel Dims.
	auto w_i = blockIdx.y;
	auto w_j = blockIdx.x;
	auto r = blockDim.x;

	auto w = iw - d_abs(w_j - r); //iw = width of input matrix
	auto h = ih - d_abs(w_i - r); //ih = height of input matrix

	//here w,h is the [dims of the submatrix].

	auto i = threadIdx.y;
	auto j = threadIdx.x;

	auto iscri = max(0, r - w_i);
	auto isrcj = max(0, r - w_j);

	//auto gscri = max(0,w_i-r);
	//auto gscrj = max(0,w_j-r);
	// if i>=0 && i + (w_i-r) is inbound then ...
	if (iscri <= i && i < iscri + w
			&& isrcj <= j && j < isrcj + h) { //i,j are within submatrix range

		auto i_idx = idx(i, j, iw);
		auto g_idx = idx(i + w_i - r, j + w_j - r, iw);
		//gradient width is same as I because
		//in the current mode of convolution, the two are the same.

		dW[idx(w_i, w_j, r)] += I[i_idx] * _G[g_idx];
	}
}

void deconvolve(Matrix& I, Matrix& _G, Matrix& dW) {
	Matrix tmp(I.size()); //make this static, somehow?

	auto s = dW.size().w;
	auto r = dW.size().w / 2; //assume square kernel, odd-size

	dim3 gridDims(I.size().w, I.size().h, 1); //--> executed throughout I
	dim3 blockDims(s, s, 1); //--> executed for each pt in kernel

	deconvolve<<<gridDims, blockDims>>>(I.d_data(), _G.d_data(), dW.d_data(),
			I.size().h, I.size().w); //G-begin index

}

ConvolutionLayer::ConvolutionLayer(int d_out) : //TODO : accept kernel size
		d_out(d_out) {
	connection = nullptr;
	d_in = 0;
}

ConvolutionLayer::~ConvolutionLayer() {
	for (int i = 0; i < d_in; ++i) {
		delete connection[i];
	}
	delete[] connection;
}

void ConvolutionLayer::setup(Size& _s, int& _d) {
	//_d = depth of input

	s = _s;
	d_in = _d;

	for (int i = 0; i < d_in; ++i) {
		I.push_back(Matrix(s));
	}

	for (int o = 0; o < d_out; ++o) {
		O.push_back(Matrix(s)); //same size

		W.push_back(Matrix::rand(5, 5)); //5,5 = kernel size
		//namedPrint(W[o]);
		//Bias depends on output matrix size,
		//Which is equivalent to the input matrix size (in case of this convolution)
		dW.push_back(Matrix::zeros(5, 5));
		dW_p.push_back(Matrix::zeros(5, 5)); //previous dW

		G.push_back(Matrix::zeros(s));
		B.push_back(Matrix::zeros(s));
		dB.push_back(Matrix::zeros(s));
		dB_p.push_back(Matrix::zeros(s)); //previous dB
	}

	connection = new bool*[d_out];

	for (int o = 0; o < d_out; ++o) {
		connection[o] = new bool[d_in];
		for (int i = 0; i < d_in; ++i) {
			connection[o][i] = true;
			//connection[o][i] = ((o % 3) == (i % 3));
			//partial connection
		}
	}
	_s = s; //same size, at least for current convolution function.
	_d = d_out;
}

std::vector<Matrix>& ConvolutionLayer::FF(std::vector<Matrix>& _I) {

	for (int i = 0; i < d_in; ++i) {
		_I[i].copyTo(I[i]);
	}

	Matrix tmp = Matrix(W[0].size());

	for (int o = 0; o < d_out; ++o) {
		O[o].zero(); //set to zero
		for (int i = 0; i < d_in; ++i) {
			if (connection[o][i]) {
				//TODO : this seems like it can be parallelized, like ""per each output layer...""
				convolve(I[i], W[o], tmp);
				O[o] += tmp;
			}
		}
		O[o] += B[o]; //add bias
	}

	return O;
}

std::vector<Matrix>& ConvolutionLayer::BP(std::vector<Matrix>& _G) {
	auto iw = s.w;
	auto ih = s.h;

	auto fw = W[0].size().w; //kernel size
	auto fh = W[0].size().h;

	auto fwr = fw / 2; //kernel size
	auto fhr = fh / 2;

	for (int i = 0; i < d_in; ++i) {
		G[i].zero(); //reset to 0
	}

	Matrix dG(G[0].size()); //TODO : make this static?
	Matrix ddW(dW[0].size()); //there are acculumants.

	for (int o = 0; o < d_out; ++o) { //for each output channel(depth):
		dW[o].zero();

		correlate(_G[o], W[o], dG); //TODO:check correlation works

		for (int i = 0; i < d_in; ++i) { //for each input channel
			if (connection[o][i]) { //if the channels are related..
				G[i] += dG;
				deconvolve(I[i], _G[o], dW[o]);
			}
		}

		dW[o] = (dW_p[o] * MOMENTUM) + (dW[o] * ETA) //no "gain" implemented yet
		- (W[o] * DECAY);

		dB[o] = (dB_p[o] * MOMENTUM) + (G[o] * ETA) //bias = gradient
		- (B[o] * DECAY);

		dW[o].copyTo(dW_p[o]);
		dB[o].copyTo(dB_p[o]);

		//dW_p[o] = dW[o];
		//dB_p[o] = dB[o];
	}
	return G;
}

void ConvolutionLayer::update() {
	for (int o = 0; o < d_out; ++o) {
		W[o] += dW[o];
		B[o] += dB[o];
	}
}
