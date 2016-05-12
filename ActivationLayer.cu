#include "ActivationLayer.h"
#include "Utility.h"

double __device__ sigmoid(double x) {
	//can only be called from device
	return 1.0 / (1.0 + exp(-x));
}

double __device__ sigmoidPrime(double x) {
	x = sigmoid(x);
	return x * (1 - x);
}

double __device__ softplus(double x) {
	return log(1 + exp(x));
}

double __device__ softplusPrime(double x) {
	return sigmoid(x);
}
double __device__ ReLU(double x) {
	return x > 0 ? x : 0;
}
double __device__ ReLUPrime(double x) {
	return x > 0 ? 1 : 0;
}

double __device__ mytanh(double x) {
	//in order to enforce device function ptr.
	return tanh(x);
}

double __device__ tanhPrime(double x) {
	x = tanh(x);
	return 1 - x * x;
	//return x * (1-x);
}
void __global__ activate(double* I, double* O, dfun f) {
	//can be called from host
	int i = threadIdx.x;
	O[i] = f(I[i]);
}
void activate(Matrix& I, Matrix& O, dfun f) {
	int n_elem = I.size().wh;
	activate<<<1, n_elem>>>
			(I.d_data(), O.d_data(), f);
	//TODO: potentially divide up to more threads?
}

__device__ dfun pf_sig = sigmoid;
__device__ dfun pf_sig_d = sigmoidPrime;
__device__ dfun pf_sp = softplus;
__device__ dfun pf_sp_d = softplusPrime;
__device__ dfun pf_relu = ReLU;
__device__ dfun pf_relu_d = ReLUPrime;
__device__ dfun pf_tanh = mytanh;
__device__ dfun pf_tanh_d = tanhPrime;

ActivationLayer::ActivationLayer(std::string _f) {


	for (auto& c : _f) {
		c = std::tolower(c);
	}

	//cudaMalloc(&f,sizeof(dfun));
	//cudaMalloc(&f_d,sizeof(dfun));

	if (_f == "sigmoid") {
		cudaMemcpyFromSymbol(&f, pf_sig, sizeof(dfun),0,cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&f_d, pf_sig_d, sizeof(dfun),0,cudaMemcpyDeviceToHost);
		//pf = sigmoid;
		//pf_d = sigmoidPrime;
		//cudaMemcpyFromSymbol(&h_f, sigmoid, sizeof(dfun));
		//cudaMemcpyFromSymbol(&h_f_d, sigmoidPrime, sizeof(dfun));
		//f = sigmoid;
		//f_d = sigmoidPrime;
	} else if (_f == "softplus") {
		cudaMemcpyFromSymbol(&f, pf_sp, sizeof(dfun),0,cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&f_d, pf_sp_d, sizeof(dfun),0,cudaMemcpyDeviceToHost);

		//pf = softplus;
		//pf_d = softplusPrime;
		//cudaMemcpyFromSymbol(&h_f, softplus, sizeof(dfun));
		//cudaMemcpyFromSymbol(&h_f_d, softplusPrime, sizeof(dfun));

		//f = softplus;
		//f_d = softplusPrime;
	} else if (_f == "relu") {
		cudaMemcpyFromSymbol(&f, pf_relu, sizeof(dfun),0,cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&f_d, pf_relu_d, sizeof(dfun),0,cudaMemcpyDeviceToHost);

		//pf = ReLU;
		//pf_d = ReLUPrime;
		//cudaMemcpyFromSymbol(&h_f, ReLU, sizeof(dfun));
		//cudaMemcpyFromSymbol(&h_f_d, ReLUPrime, sizeof(dfun));
		//f = ReLU;
		//f_d = ReLUPrime;
	} else if (_f == "tanh") {
		cudaMemcpyFromSymbol(&f, pf_tanh, sizeof(dfun),0,cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&f_d, pf_tanh_d, sizeof(dfun),0,cudaMemcpyDeviceToHost);

		//pf = mytanh;
		//pf_d = tanhPrime;
		//cudaMemcpyFromSymbol(&h_f, mytanh, sizeof(dfun));
		//cudaMemcpyFromSymbol(&h_f_d, tanhPrime, sizeof(dfun));
		//f = mytanh;
		//f_d = tanhPrime;
	} else {
		throw "WRONG ACTIVATION FUNCTION!!";
	}
	//dfun h_f; // = (dfun*)malloc(sizeof(dfun));
	//dfun h_f_d;// = (dfun*)malloc(sizeof(dfun));
	//cudaMemcpyFromSymbol(&f, pf, sizeof(dfun),0,cudaMemcpyDeviceToHost);
	//cudaMemcpyFromSymbol(&f_d, pf_d, sizeof(dfun),0,cudaMemcpyDeviceToHost);

	//cudaMemcpy(f,&h_f,sizeof(dfun),cudaMemcpyHostToDevice);
	//cudaMemcpy(f_d,&h_f_d,sizeof(dfun),cudaMemcpyHostToDevice);

}

ActivationLayer::~ActivationLayer(){
	//TODO : find out if freeing is necessary (there's probably reference)
	//cudaFree(f);
	//cudaFree(f_d);
}

void ActivationLayer::setup(Size& _s, int& _d) {
	s = _s;
	d = _d;

	for (int i = 0; i < d; ++i) {
		I.push_back(Matrix(s));
		G.push_back(Matrix(s));
		O.push_back(Matrix(s));
	}

}

std::vector<Matrix>& ActivationLayer::FF(std::vector<Matrix>& _I) {
	for (int i = 0; i < d; ++i) {
		_I[i].copyTo(I[i]);
		activate(I[i], O[i], f);
	}
	return O;
}

std::vector<Matrix>& ActivationLayer::BP(std::vector<Matrix>& _G) {
	Matrix tmp(s);
	for (int i = 0; i < d; ++i) {
		activate(I[i], tmp, f_d);
		G[i] = _G[i] % tmp;
		//or consider setting G[i].dat as destination of mul.
	}
	return G;
}


void ActivationLayer::update(){

}
