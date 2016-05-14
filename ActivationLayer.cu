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
void __global__ sigmoid(double* I, double* O){
	int i = threadIdx.x;
	O[i]  = 1.0 / (1.0 + exp(-I[i]));
}
void __global__ activate(double* I, double* O, dfun f) {
	//can be called from host
	int i = threadIdx.x;
	O[i] = f(I[i]);
}
void __global__ activate(double* I, double* O, dfun f, int lim) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i<lim)
		O[i] = f(I[i]);
}
void activate(Matrix& I, Matrix& O, dfun f) {

	int n_elem = I.size().wh;
	//hline();
	//namedPrint(I);

	if(n_elem < 1024){
		activate<<<1, n_elem>>>
					(I.d_data(), O.d_data(), f);
	}else{
		activate<<< (n_elem+255) / 256, 256>>>
					(I.d_data(), O.d_data(), f, n_elem);
	}
	//O.set_sync(false);
	//namedPrint(O);
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

	if (_f == "sigmoid") {
		cudaMemcpyFromSymbol(&f, pf_sig, sizeof(dfun));
		cudaMemcpyFromSymbol(&f_d, pf_sig_d, sizeof(dfun));
	} else if (_f == "softplus") {
		cudaMemcpyFromSymbol(&f, pf_sp, sizeof(dfun));
		cudaMemcpyFromSymbol(&f_d, pf_sp_d, sizeof(dfun));
	} else if (_f == "relu") {
		cudaMemcpyFromSymbol(&f, pf_relu, sizeof(dfun));
		cudaMemcpyFromSymbol(&f_d, pf_relu_d, sizeof(dfun));
	} else if (_f == "tanh") {
		cudaMemcpyFromSymbol(&f, pf_tanh, sizeof(dfun));
		cudaMemcpyFromSymbol(&f_d, pf_tanh_d, sizeof(dfun));
	} else {
		throw "WRONG ACTIVATION FUNCTION!!";
	}

}

ActivationLayer::~ActivationLayer(){

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
		//namedPrint(I[i]);
		//sigmoid<<<1,s.wh>>>(I[i].d_data(),O[i].d_data());
		activate(I[i], O[i], f);
		O[i].set_sync(false); //O[i] is not synced anymore!
		//namedPrint(O[i]);

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
