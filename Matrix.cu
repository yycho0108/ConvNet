/*
 * Matrix.cpp
 *
 *  Created on: May 7, 2016
 *      Author: jamiecho
 */

#include "Matrix.h"
#include "curand.h"
#include "curand_kernel.h"
#include <cassert>

// TODO : optimize for non in-place calculations by setting output ptrs

std::ostream& operator<<(std::ostream& os, Matrix& m){
	m.print(os);
	return os;
}

bool isNaN(Matrix& m){
	m.sync();
	for(int i=0;i<m.size().wh;++i){
		if(isnan(m.data()[i]))
			return true;
	}
	return false;
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
__global__ void apply(double* I, dfun f){
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

__global__ void dotT(double* a, double* b, double* o, int com, int w, int h){
	//b needs to be transposed prior to this.
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	/*
	 *  v1a v1b v1c     w1a w2a w3a	     v1w1  v1w2 v1w3
	 *  v2a v2b v2c  *  w1b w2b w3b ---> v2w1  v2w2 v2w3
	 *  v3a v3b v3c     w1c w2c w3c      v3w1  v3w2 v3w3
	 */
	if(i<h && j<w)
		o[idx(i,j,w)] = vdot(a + i*com, b+j*com, com); //length of common.
	// here a = mat of n x com
	// b = mat of com x m
	// c = mat of n x m
}

Matrix dot(const Matrix& a, const Matrix& b){
	////assert(a.size().w == b.size().h);
	int com = a.size().w; // == b.size().h;
	Matrix bT = Matrix::transpose(b);
	Matrix o(b.size().w, a.size().h);

	auto s = o.size();

	if(s.wh < 1024){
		dim3 blockDims(s.w, s.h);
		dotT<<<1,blockDims>>>(a.d_data(),bT.d_data(),o.d_data(),com);
	}else{
		dim3 blockDims(16,16);
		dim3 gridDims((s.w+15)/16,(s.h+15)/16);
		dotT<<<gridDims,blockDims>>>(a.d_data(),bT.d_data(),o.d_data(),com, s.w, s.h);
	}

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

__global__ void _transpose(double* I, double* O, int w, int h){
	//TODO : optimize with 'shared memory'
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if(i<h && j<w)
		O[idx(j,i,h)] = I[idx(i,j,w)];
}

__global__ void _lt(double* I, double t, double* O, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		O[i] = (I[i] < t);
}
__global__ void _le(double* I, double t, double* O, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		O[i] = (I[i] <= t);
}

__global__ void _gt(double* I, double t, double* O, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		O[i] = (I[i] > t);
}

__global__ void _ge(double* I, double t, double* O, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		O[i] = (I[i] >= t);
}

__global__ void _eq(double* I, double t, double* O, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		O[i] = (I[i] == t);
}

Matrix::Matrix():d_dat(nullptr),dat(nullptr),s(0,0),synced(false){
	//nothing!
}

Matrix::Matrix(Size s, double* d)
:Matrix(s.w,s.h,d){

}

Matrix::Matrix(int w, int h, double* d)
:s(w,h){
	//d is host pointer

	int sz = w*h*sizeof(double);

	dat = (double*) malloc(sz);
	cudaMalloc(&d_dat,sz);

	if(d != nullptr){
		memcpy(dat,d,sz);
		cudaMemcpy(d_dat,d,sz,cudaMemcpyHostToDevice);
	}
	synced = true;
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
	int sz = o.s.wh * sizeof(double);

	if(s.wh == o.s.wh){
		//no need to reallocate...
		s = o.s; //reset size
	}else{
		cudaFree(d_dat);
		free(dat);
		cudaMalloc(&d_dat,sz);
		dat = (double*) malloc(sz);
	}

	cudaMemcpy(d_dat,o.d_dat,sz,cudaMemcpyDeviceToDevice);
	memcpy(dat,o.dat,sz);
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

Matrix& Matrix::operator+=(const Matrix& o){
	//assert(size() == o.size());
	add(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}
Matrix& Matrix::operator-=(const Matrix& o){
	//assert(size() == o.size());
	sub(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix& Matrix::operator/=(const Matrix& o){
	//assert(size() == o.size());
	div(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}
Matrix& Matrix::operator%=(const Matrix& o){
	//assert(size() == o.size());
	mul(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}

Matrix& Matrix::operator+=(const Matrix&& o){
	//assert(size() == o.size());
	add(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}
Matrix& Matrix::operator-=(const Matrix&& o){
	//assert(size() == o.size());
	sub(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}


Matrix& Matrix::operator/=(const Matrix&& o){
	//assert(size() == o.size());
	div(d_dat,o.d_dat,d_dat,s.wh);
	synced = false;
	return *this;
}
Matrix& Matrix::operator%=(const Matrix&& o){
	//assert(size() == o.size());
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

Matrix Matrix::operator+(const Matrix& o) const{
	//assert(size() == o.size());
	Matrix m;
	copyTo(m);
	return m += o;
}
Matrix Matrix::operator-(const Matrix& o) const{
	//assert(size() == o.size());
	Matrix m;
	copyTo(m);
	return m -= o;
}
Matrix Matrix::operator*(const Matrix& o) const{
	return dot(*this,o);
}
Matrix Matrix::operator/(const Matrix& o) const{
	//assert(size() == o.size());
	Matrix m;
	copyTo(m);
	return m /= o;
}
Matrix Matrix::operator%(const Matrix& o) const{
	//assert(size() == o.size());
	Matrix m;
	copyTo(m);
	return m %= o;
}
Matrix Matrix::operator+(const Matrix&& o) const{
	//assert(size() == o.size());
	Matrix m;
	copyTo(m);
	return m += o;
}
Matrix Matrix::operator-(const Matrix&& o) const{
	//assert(size() == o.size());
	Matrix m;
	copyTo(m);
	return m -= o;
}
Matrix Matrix::operator*(const Matrix&& o) const{
	return dot(*this,o);
}
Matrix Matrix::operator/(const Matrix&& o) const{
	//assert(size() == o.size());
	Matrix m;
	copyTo(m);
	return m /= o;
}
Matrix Matrix::operator%(const Matrix&& o) const{
	//assert(size() == o.size());
	Matrix m;
	copyTo(m);
	return m %= o;
}

Matrix Matrix::operator+(double o) const{
	Matrix m;
	copyTo(m);
	return m += o;
}

Matrix Matrix::operator-(double o) const{
	Matrix m;
	copyTo(m);
	return m -= o;
}

Matrix Matrix::operator*(double o) const{
	Matrix m;
	copyTo(m);
	return m *= o;
}

Matrix Matrix::operator/(double o) const{
	Matrix m;
	copyTo(m);
	return m /= o;
}

Matrix Matrix::operator-() const{
	Matrix m;
	copyTo(m);
	return m *= -1.0;
}

Matrix Matrix::operator<(double val) const{
	Matrix m(this->s);
	int nb = (s.wh + 255) / 256; //# of blocks
	_lt<<<nb,256>>>(d_dat,val,m.d_dat,s.wh);
	return m;
}
Matrix Matrix::operator<=(double val) const{
	Matrix m(this->s);
	int nb = (s.wh + 255) / 256; //# of blocks
	_le<<<nb,256>>>(d_dat,val,m.d_dat,s.wh);
	return m;
}
Matrix Matrix::operator>(double val) const{
	Matrix m(this->s);
	int nb = (s.wh + 255) / 256; //# of blocks
	_gt<<<nb,256>>>(d_dat,val,m.d_dat,s.wh);
	return m;
}
Matrix Matrix::operator>=(double val) const{
	Matrix m(this->s);
	int nb = (s.wh + 255) / 256; //# of blocks
	_ge<<<nb,256>>>(d_dat,val,m.d_dat,s.wh);
	return m;
}
Matrix Matrix::operator==(double val) const{
	// TODO : give small buffer for equality(eps)?
	Matrix m(this->s);
	int nb = (s.wh + 255) / 256; //# of blocks
	_eq<<<nb,256>>>(d_dat,val,m.d_dat,s.wh);
	return m;
}

double Matrix::operator()(int i, int j){
	sync();
	return dat[idx(i,j,s.w)];
}


Matrix& Matrix::apply(dfun f){
	dfun f_d; //must be device function... I think.
	cudaMemcpyFromSymbol(&f_d,f,sizeof(dfun));
	::apply<<<1,s.wh>>>(d_dat,f_d);
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
	rnd.randn(d_dat,s.wh,0.0,0.1);
	synced = false;
}

void Matrix::randn(double mean, double stddev){
	rnd.randn(d_dat,s.wh,mean,stddev);

	//rnd.rand(d_dat,s.wh); // 0 ~ 1
	//*this -= 0.5;
	synced = false;
}

void Matrix::randu(double min, double max){
	rnd.randu(d_dat,s.wh);

	if(min != 0.0 || max != 1.0){
		*this *= (max-min);
		*this -= min;
	}

	//rnd.rand(d_dat,s.wh); // 0 ~ 1
	//*this -= 0.5;
	synced = false;
}
void Matrix::abs(){
	::abs(d_dat,d_dat,s.wh);
}
void Matrix::copyTo(Matrix& m, cudaStream_t* stream) const{
	auto sz = s.wh*sizeof(double);

	if(m.d_dat == nullptr || size() != m.size()){
		cudaFree(m.d_dat);
		free(m.dat);
		cudaMalloc(&m.d_dat,sz);
		m.dat = (double*) malloc(sz);
	}

	m.s = s;
	memcpy(m.dat,dat,sz);

	if(stream){
		cudaMemcpyAsync(m.d_dat,d_dat,sz,cudaMemcpyDeviceToDevice,*stream);
	}else{
		cudaMemcpy(m.d_dat,d_dat,sz,cudaMemcpyDeviceToDevice);
	}

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

Matrix Matrix::transpose(const Matrix& I){
	//TODO : Improve transposition logic
	auto s = I.size();

	Matrix O(s.h,s.w);

	if(s.wh < 1024){
		dim3 blockDims(s.w,s.h);
		_transpose<<<1,blockDims>>>(I.d_dat,O.d_dat);
	}else{
		//split into approx. evenly sized blocks of 16x16 (or else, but determine that later)
		int nb = (s.wh + 255)/256; // total number of blocks
		dim3 blockDims(16,16);
		dim3 gridDims((s.w+15)/16,(s.h+15)/16);
		_transpose<<<gridDims,blockDims>>>(I.d_dat,O.d_dat,s.w,s.h);

		//dim3 blockDims(TILE_DIM,TILE_DIM);
		//dim3 gridDims((s.w + TILE_DIM -1)/TILE_DIM, (s.h + TILE_DIM - 1)/TILE_DIM);
		//_transposeCoalesced<<<gridDims,blockDims>>>(I.d_dat,O.d_dat);
	}
	return O;
}
Matrix Matrix::abs(const Matrix& src){
	Matrix dst;
	src.copyTo(dst);
	dst.abs();
	return dst;
}

void Matrix::sync(){
	if(!synced){
		cudaMemcpy(dat,d_dat,s.wh*sizeof(double),cudaMemcpyDeviceToHost);
		synced = true;
	}
}
void Matrix::sync_r(){
	cudaMemcpy(d_dat,dat,s.wh*sizeof(double),cudaMemcpyHostToDevice);
	synced = true;
}

void Matrix::set_sync(bool s){
	synced = s;
}

void Matrix::print(std::ostream& out){
	sync();
	int h = s.h;
	int w = s.w;
	for(int i=0;i<h;++i){
		for(int j=0;j<w;++j){
			out << dat[idx(i,j,w)] << ' ';
		}
		out << '\n';
	}
	out << std::endl;
}
Size Matrix::size() const{
	return s;
}

double* Matrix::data() const{
	return dat;
	//host data (cpu);
}

double* Matrix::d_data() const{
	return d_dat;
	//device data (gpu)
}

Matrix operator+(double v, const Matrix& m){
	return m + v;
}

Matrix operator-(double v, const Matrix& m){
	//TODO : specialize? I feel like mul is efficient enough... well.
	return (m-v) * -1.0;
}
Matrix operator*(double v, const Matrix& m){
	return m*v;

}
/*
Matrix operator/(double v, const Matrix& m){
	//TODO : implement
	return m *= (1.0/v);
}
*/

Matrix&& operator+(Matrix&& a, const Matrix& b){
	return std::move(a += b);
}
Matrix&& operator-(Matrix&& a, const Matrix& b){
	return std::move(a -= b);
}
Matrix&& operator/(Matrix&& a, const Matrix& b){
	return std::move(a /= b);
}
Matrix&& operator%(Matrix&& a, const Matrix& b){
	return std::move(a %= b);
}

Matrix&& operator+(Matrix&& a, const Matrix&& b){
	return std::move(a += b);
}
Matrix&& operator-(Matrix&& a, const Matrix&& b){
	return std::move(a -= b);
}
Matrix&& operator/(Matrix&& a, const Matrix&& b){
	return std::move(a /= b);
}
Matrix&& operator%(Matrix&& a, const Matrix&& b){
	return std::move(a %= b);
}


Matrix&& operator+(Matrix&& m, double v){
	return std::move(m += v);
}

Matrix&& operator-(Matrix&& m, double v){
	return std::move(m -= v);
}

Matrix&& operator*(Matrix&& m, double v){
	return std::move(m *= v);
}
Matrix&& operator/(Matrix&& m, double v){
	return std::move(m /= v);
}
Matrix&& operator+(double v, Matrix&& m){
	return std::move(m += v);
}
Matrix&& operator-(double v, Matrix&& m){
	m -= v;
	return std::move(-m);
}
Matrix&& operator*(double v, Matrix&& m){
	return std::move(m *= v);
}
/*
Matrix&& operator/(double v, Matrix&& m){
	return m *= (1.0/v);
}
*/

Matrix&& operator-(Matrix&& m){
	return std::move(m *= -1.0);
}
