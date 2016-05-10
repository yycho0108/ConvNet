/*
 * Matrix.cpp
 *
 *  Created on: May 7, 2016
 *      Author: jamiecho
 */

#include "Matrix.h"
#include <functional>

using dfun = double (*)(double);

Matrix::Matrix(Size s, double* d)
:Matrix(s.n, s.m, d){
}
Matrix::Matrix(int n, int m, double* d) {
	int sz = n*m*sizeof(double);
	dat = (double*) malloc(sz);
	cudaMalloc(&d_dat,sz);

	if(d != nullptr){
		cudaMemcpy(d_dat,d,sz,cudaMemcpyHostToDevice);
	}
}

Matrix::Matrix(Matrix& m){
	//copy constructor
	s = m.s;
	int sz = s.n * s.m * sizeof(double);

	cudaMalloc(&d_dat, sz);
	cudaMemcpy(d_dat,m.d_dat,sz,cudaMemcpyDeviceToDevice);
}
Matrix::Matrix(Matrix&& m){
	//move constructor
	s = m.s;
	dat = m.dat;
	d_dat = m.d_dat;
	m.dat = nullptr;
	m.d_dat = nullptr;
}
Matrix::~Matrix() {
	free(dat);
	cudaFree(d_dat);
	// TODO Auto-generated destructor stub
}

void Matrix::sync(){
	cudaMemcpy(dat,d_dat,s.n*s.m*sizeof(double),cudaMemcpyDeviceToHost);
}

__global__ void apply(double* I, dfun f){
	int i = threadIdx.x;
	I[i] = f(I[i]);

}

Matrix& Matrix::apply(dfun f){
	dfun f_d; //device function
	cudaMemcpyFromSymbol(&f_d,f,sizeof(dfun));
	apply(d_dat,f_d);
	//if 'device function' trick doesn't work, copy function to symbol with
	//cudaMemcpyFromSymbol( &h_f[0], pfunc1, sizeof(func));
	//or equivalent syntax.
	return *this;
}

void Matrix::zero(){
	cudaMemset(d_dat,0,s.n*s.m*sizeof(double));
}

//Matrix& Matrix::apply(std::function<double(double)> f){
//
//	return *this;
//}
