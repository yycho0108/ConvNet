/*
 * Matrix.cpp
 *
 *  Created on: May 7, 2016
 *      Author: jamiecho
 */

#include "Matrix.h"
#include "curand.h"
#include "curand_kernel.h"

std::ostream& operator<<(std::ostream& os, Matrix& m){
	m.print(os);
	return os;
}

typedef double (*dfun)(double);
RandManager Matrix::rnd = RandManager(); //or some smaller value? welp.

__device__ double vdot(double* a, double* b, int n){ //dot product of two vectors.
	double res = 0;
	for(int i=0;i<n;++i){
		res += a[i]*b[i];
	}
	return res;
}
__global__ void _apply(double* I, dfun f){
	int i = threadIdx.x;
	I[i] = f(I[i]);
}

__global__ void dotT(double* a, double* b, double* o, int com){
	//b needs to be transposed prior to this.
	auto h = blockDim.y;
	auto w = blockDim.x;

	auto i = threadIdx.y;
	auto j = threadIdx.x;

	/*
	 *  v1a v1b v1c     w1a w2a w3a	     v1w1  v1w2 v1w3
	 *  v2a v2b v2c  *  w1b w2b w3b ---> v2w1  v2w2 v2w3
	 *  v3a v3b v3c     w1c w2c w3c      v3w1  v3w2 v3w3
	 */
	o[idx(i,j,w)] = vdot(a + i*com, b+j*com, com); //length of common.
	// here a = mat of n x com
	// b = mat of com x m
	// c = mat of n x m
}

Matrix dot(Matrix& a, Matrix& b){
	int com = a.size().w; // == b.size().h;
	Matrix bT = Matrix::transpose(b);
	Matrix o(b.size().w, a.size().h);

	dim3 blockDims(b.size().w, a.size().h);
	dotT<<<1,blockDims>>>(a.d_data(),bT.d_data(),o.d_data(),com);

	return o;
}

/*__global__ void dot(double* a, double* b, double* o,
		int aw, int bw){
	auto i = threadIdx.y;
	auto j = threadIdx.x;

}
*/

__global__ void _eye(double* d_dat, int w){
	auto i = threadIdx.x;
	d_dat[idx(i,i,w)] = 1.0;
}

__global__ void _transpose(double* I, double* O){
	int h = blockDim.y;
	int w = blockDim.x;
	int i = threadIdx.y;
	int j = threadIdx.x;
	O[idx(j,i,h)] = I[idx(i,j,w)];
}

Matrix::Matrix():d_dat(nullptr),dat(nullptr),s(0,0),synced(false){
	//nothing!
}

Matrix::Matrix(Size s, double* d)
:Matrix(s.w,s.h,d){

}

Matrix::Matrix(int w, int h, double* d)
:s(w,h){

	int sz = w*h*sizeof(double);

	dat = (double*) malloc(sz);
	cudaMalloc(&d_dat,sz);

	if(d != nullptr){
		memcpy(dat,d,sz);
		cudaMemcpy(d_dat,d,sz,cudaMemcpyHostToDevice);
	}
	synced = false;
}

Matrix::Matrix(const Matrix& m){
	//copy constructor
	s = m.s;
	int sz = s.wh * sizeof(double);

	cudaMalloc(&d_dat, sz);
	cudaMemcpy(d_dat,m.d_dat,sz,cudaMemcpyDeviceToDevice);

	dat = (double*) malloc(sz);
	memcpy(dat,m.dat,sz);

	synced = m.synced;
}

Matrix::Matrix(Matrix&& o){
	//move constructor

	s = o.s;
	dat = o.dat;
	d_dat = o.d_dat;
	o.dat = nullptr;
	o.d_dat = nullptr;

	synced = o.synced;
}

Matrix::~Matrix() {
	free(dat);
	cudaFree(d_dat);
}

Matrix& Matrix::Matrix::operator=(const Matrix& o){
	throw "Don't Come HERE!!";
	/* problematic for memory management*/
	s = o.s;
	dat = o.dat;
	d_dat = o.d_dat;

	synced = o.synced;
	return *this;
}

Matrix& Matrix::Matrix::operator=(Matrix&& o){
	free(dat);
	cudaFree(d_dat);

	s = o.s;
	dat = o.dat;
	d_dat = o.d_dat;

	o.dat = nullptr;
	o.d_dat = nullptr;
	synced = o.synced;

	return *this;
}

Matrix& Matrix::operator+=(Matrix& o){
	add(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}
Matrix& Matrix::operator-=(Matrix& o){
	sub(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix& Matrix::operator/=(Matrix& o){
	div(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}
Matrix& Matrix::operator%=(Matrix& o){
	mul(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix& Matrix::operator+=(Matrix&& o){
	add(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}
Matrix& Matrix::operator-=(Matrix&& o){
	sub(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}


Matrix& Matrix::operator/=(Matrix&& o){
	div(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}
Matrix& Matrix::operator%=(Matrix&& o){
	mul(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix& Matrix::operator+=(double o){
	add(d_dat,o,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix& Matrix::operator-=(double o){
	sub(d_dat,o,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix& Matrix::operator*=(double o){
	mul(d_dat,o,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix& Matrix::operator/=(double o){
	div(d_dat,o,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix Matrix::operator+(Matrix& o){
	Matrix m;
	copyTo(m);
	return m += o;
}
Matrix Matrix::operator-(Matrix& o){
	Matrix m;
	copyTo(m);
	return m -= o;
}
Matrix Matrix::operator*(Matrix& o){
	return dot(*this,o);
}
Matrix Matrix::operator/(Matrix& o){
	Matrix m;
	copyTo(m);
	return m /= o;
}
Matrix Matrix::operator%(Matrix& o){
	Matrix m;
	copyTo(m);
	return m %= o;
}
Matrix Matrix::operator+(Matrix&& o){
	Matrix m;
	copyTo(m);
	return m += o;
}
Matrix Matrix::operator-(Matrix&& o){
	Matrix m;
	copyTo(m);
	return m -= o;
}
Matrix Matrix::operator*(Matrix&& o){
	return dot(*this,o);
}
Matrix Matrix::operator/(Matrix&& o){
	Matrix m;
	copyTo(m);
	return m /= o;
}
Matrix Matrix::operator%(Matrix&& o){
	Matrix m;
	copyTo(m);
	return m %= o;
}

Matrix Matrix::operator+(double o){
	Matrix m;
	copyTo(m);
	return m += o;
}

Matrix Matrix::operator-(double o){
	Matrix m;
	copyTo(m);
	return m -= o;
}

Matrix Matrix::operator*(double o){
	Matrix m;
	copyTo(m);
	return m *= o;
}

Matrix Matrix::operator/(double o){
	Matrix m;
	copyTo(m);
	return m /= o;
}

double Matrix::operator()(int i, int j){
	sync();
	return dat[idx(i,j,s.w)];
}


Matrix& Matrix::apply(dfun f){
	dfun f_d; //must be device function... I think.
	cudaMemcpyFromSymbol(&f_d,f,sizeof(dfun));
	_apply<<<1,s.wh>>>(d_dat,f_d);
	synced = false;
	//if 'device function' trick doesn't work, copy function to symbol with
	//cudaMemcpyFromSymbol( &h_f[0], pfunc1, sizeof(func));
	//or equivalent syntax.
	return *this;
}


double Matrix::max(Size* idx){
	sync();
	double res = -99999.0;
	for(int i=0;i<s.wh;++i){
		if(dat[i] > res){
			res = dat[i];

			if(idx){
				idx->h = i / s.w;
				idx->w = i % s.w;
			}

		}
	}
	return res;
	//max of all elem
}

double Matrix::min(Size* idx){
	sync();
	double res = 99999.0;
	for(int i=0;i<s.wh;++i){
		if(dat[i] < res){
			res = dat[i];

			if(idx){
				idx->h = i / s.w;
				idx->w = i % s.w;
			}
		}
	}
	return res;
	//max of all elem
}

double Matrix::sum(){
	sync();
	double res=0;
	for(int i=0;i<s.wh;++i){
		res += dat[i];
	}
	return res;
	//sum of all elem
}
double Matrix::avg(){
	return sum() / s.wh;
	//avg of all elem
}

void Matrix::zero(){
	auto sz = s.wh*sizeof(double);
	cudaMemset(d_dat,0,sz);
	synced = false;
}
void Matrix::one(){
	zero();
	*this += 1.0;
	synced = false;
}
void Matrix::eye(){
	zero();
	int n = s.w<s.h?s.w:s.h; //w-h

	_eye<<<1,n>>>(d_dat,s.w);

	synced = false;
}
void Matrix::rand(){
	rnd.rand(d_dat,s.wh);
	synced = false;
}
void Matrix::copyTo(Matrix& m){
	auto sz = s.wh*sizeof(double);

	if(m.d_dat == nullptr || size() != m.size()){
		cudaFree(m.d_dat);
		free(m.dat);
		cudaMalloc(&m.d_dat,sz);
		m.dat = (double*) malloc(sz);
	}

	m.s = s;
	memcpy(m.dat,dat,sz);
	cudaMemcpy(m.d_dat,d_dat,sz,cudaMemcpyDeviceToDevice);
	m.synced = synced;
}

void Matrix::transpose(){
	//TODO : figure out a better way to do this.s
	*this = transpose(*this);
	synced = false;
	//since transpose outputs an rvalue,
	//it automatically deallocates my resources
	//and moves the transposed result into the matrix.
}

//static functions

Matrix Matrix::eye(Size s){
	return Matrix::eye(s.w,s.h);
}
Matrix Matrix::eye(int w, int h){
	Matrix m(w,h);
	m.eye();
	return m;
}

Matrix Matrix::zeros(Size s){
	return Matrix::zeros(s.w,s.h);
}

Matrix Matrix::zeros(int w, int h){
	Matrix m(w,h);
	m.zero();
	return m;
}

Matrix Matrix::ones(Size s){
	return Matrix::ones(s.w,s.h);
}

Matrix Matrix::ones(int w, int h){
	Matrix m(w,h);
	m.one();
	return m;
}

Matrix Matrix::rand(Size s){
	return Matrix::rand(s.w,s.h);
}

Matrix Matrix::rand(int w, int h){
	Matrix m(w,h);
	m.rand();
	return m;
}

Matrix Matrix::transpose(Matrix& I){
	//TODO : Improve transposition logic
	Matrix O(I.size());
	dim3 blockDims(I.size().w, I.size().h);
	_transpose<<<1,blockDims>>>(I.d_dat,O.d_dat);
	return O;
}

void Matrix::sync(){
	if(!synced){
		cudaMemcpy(dat,d_dat,s.wh*sizeof(double),cudaMemcpyDeviceToHost);
		synced = true;
	}
}

void Matrix::print(std::ostream& out){
	sync();
	int w = s.w;
	for(int i=0;i<s.h;++i){
		for(int j=0;j<s.w;++j){
			out << dat[idx(i,j,s.w)] << ' ';
		}
		out << '\n';
	}
	out << std::endl;
}
Size Matrix::size(){
	return s;
}

double* Matrix::data(){
	return dat;
	//host data (cpu);
}

double* Matrix::d_data(){
	return d_dat;
	//device data (gpu)
}
