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

__global__ void _convolve(double* d_i, double* d_k, double* d_o,int r){
	//assuming kernel size 3x3
	int i = threadIdx.x;
	int j = threadIdx.y;

	int n = blockDim.x;
	int m = blockDim.y;

	d_o[idx(i,j,m)] = 0;
	for(int ki=-r;ki<=r;++ki){
		for(int kj=-r;kj<=r;++kj){
			if(inbound(i+ki,j+kj,n,m)){
				d_o[idx(i,j,m)] +=
					d_i[idx(i+ki,j+kj,m)]
					* d_k[idx(ki+r,kj+r,r)]; //flip here if correlation
			}
			//effectively zero-padding
			//may change to VALID convolution later

			//d_o[i][j] += d_i[i+ki][j+kj] * d_k[ki+r][kj+r]
		}
	}
	//TODO : IMPLEMENT
}

void convolve_d(double* d_i, double* d_k, double* d_o,
	//if all ptrs are in gpu
		int n, int m, int r){
	dim3 g(1,1);
	dim3 b(n,m);
	_convolve<<<g,b>>>(d_i,d_k,d_o,r);
}

void convolve(double* i, double* k, double* o,
		int n, int m, int r){
	double* d_i, *d_k, *d_o;


	int sz = n*m*sizeof(double);
	int ksz = 3*3*sizeof(double);

	cudaMalloc(&d_i,sz);
	cudaMalloc(&d_k,ksz);
	cudaMalloc(&d_o,sz);

	cudaMemcpy(d_i,i,sz,cudaMemcpyHostToDevice);
	cudaMemcpy(d_k,k,ksz,cudaMemcpyHostToDevice);


	clock_t start = clock();
	convolve_d(d_i,d_k,d_o,n,m,r);

	clock_t end = clock();
	printf("Took %f Seconds", float(end-start)/CLOCKS_PER_SEC);

	cudaMemcpy(o,d_o,sz,cudaMemcpyDeviceToHost);

	cudaFree(d_i);
	cudaFree(d_k);
	cudaFree(d_o);

	return;
}
