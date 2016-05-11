#include "Utility.h"

__global__ void _add(double* a, double* b, double* out){
	out[i] = a[i]+b[i];
}
__global__ void _sub(double* a, double* b, double* out){
	out[i] = a[i]-b[i];
}
__global__ void _mul(double* a, double* b, double* out){
	out[i] = a[i]*b[i];
}
__global__ void _div(double* a, double* b, double* out){
	out[i] = a[i]/b[i];
}

__global__ void _add(double* a, double b, double* out){
	out[i] = a[i]+b;
}
__global__ void _sub(double* a, double b, double* out){
	out[i] = a[i]-b;
}
__global__ void _mul(double* a, double b, double* out){
	out[i] = a[i]*b;
}
__global__ void _div(double* a, double b, double* out){
	out[i] = a[i]/b;
}


void add(double* a, double* b, double* o, int n){
	_add<<<1,n>>>(a,b,o);
}
void sub(double* a, double* b, double* o, int n){
	_sub<<<1,n>>>(a,b,o);
}
void mul(double* a, double* b, double* o, int n){
	_mul<<<1,n>>>(a,b,o);
}
void div(double* a, double* b, double* o, int n){
	_div<<<1,n>>>(a,b,o);
}

void add(double* a, double b, double* o, int n){
	_add<<<1,n>>>(a,b,o);
}
void sub(double* a, double b, double* o, int n){
	_sub<<<1,n>>>(a,b,o);
}
void mul(double* a, double b, double* o, int n){
	_mul<<<1,n>>>(a,b,o);
}
void div(double* a, double b, double* o, int n){
	_div<<<1,n>>>(a,b,o);
}

__global__ void _convolve(double* d_i, double* d_k, double* d_o,int r){
	//assuming kernel size 3x3
	int i = threadIdx.y;
	int j = threadIdx.x;

	int h = blockDim.y;
	int w = blockDim.x;

	d_o[idx(i,j,w)] = 0;
	for(int ki=-r;ki<=r;++ki){
		for(int kj=-r;kj<=r;++kj){
			if(inbound(i+ki,j+kj,h,w)){
				d_o[idx(i,j,h)] +=
					d_i[idx(i+ki,j+kj,h)]
					* d_k[idx(ki+r,kj+r,2*r+1)]; //flip here if correlation
			}
			//effectively zero-padding
			//may change to VALID convolution later

			//d_o[i][j] += d_i[i+ki][j+kj] * d_k[ki+r][kj+r]
		}
	}
	//TODO : IMPLEMENT
}
__global__ void _correlate(double* d_i, double* d_k, double* d_o,int r){
	//assuming kernel size 3x3

	int i = threadIdx.y;
	int j = threadIdx.x;

	int h = blockDim.y;
	int w = blockDim.x;

	d_o[idx(i,j,w)] = 0;
	for(int ki=-r;ki<=r;++ki){
		for(int kj=-r;kj<=r;++kj){
			if(inbound(i+ki,j+kj,h,w)){
				d_o[idx(i,j,w)] +=
					d_i[idx(i+ki,j+kj,w)]
					* d_k[idx(r-ki,r-kj,2*r+1)]; //flip here if correlation
			}
			//effectively zero-padding
			//may change to VALID convolution later

			//d_o[i][j] += d_i[i+ki][j+kj] * d_k[ki+r][kj+r]
		}
	}
}
void convolve_d(double* d_i, double* d_k, double* d_o,
	//if all ptrs are in gpu
		int w, int h, int r){
	dim3 g(1,1);
	dim3 b(w,h);
	_convolve<<<g,b>>>(d_i,d_k,d_o,r);
}

void correlate_d(double* d_i, double* d_k, double* d_o,
	//if all ptrs are in gpu
		int w, int h, int r){
	dim3 g(1,1);
	dim3 b(w,h);
	_correlate<<<g,b>>>(d_i,d_k,d_o,r);
}

void convolve(double* i, double* k, double* o,
		int w, int h, int r){
	double* d_i, *d_k, *d_o;


	int sz = w*h*sizeof(double);
	int ksz = (2*r+1)*(2*r+1)*sizeof(double);

	cudaMalloc(&d_i,sz);
	cudaMalloc(&d_k,ksz);
	cudaMalloc(&d_o,sz);

	cudaMemcpy(d_i,i,sz,cudaMemcpyHostToDevice);
	cudaMemcpy(d_k,k,ksz,cudaMemcpyHostToDevice);


	//clock_t start = clock();
	convolve_d(d_i,d_k,d_o,w,h,r);

	//clock_t end = clock();
	//printf("Took %f Seconds", float(end-start)/CLOCKS_PER_SEC);

	cudaMemcpy(o,d_o,sz,cudaMemcpyDeviceToHost);

	cudaFree(d_i);
	cudaFree(d_k);
	cudaFree(d_o);

	return;
}
