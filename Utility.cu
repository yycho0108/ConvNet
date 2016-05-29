#include "Utility.h"

/* n < 1024 */
__global__ void _add(const double* a, const double* b, double* out){
	int i = threadIdx.x;
	out[i] = a[i]+b[i];
}
__global__ void _sub(const double* a, const double* b, double* out){
	int i = threadIdx.x;
	out[i] = a[i]-b[i];
}
__global__ void _mul(const double* a, const double* b, double* out){
	int i = threadIdx.x;
	out[i] = a[i]*b[i];
}
__global__ void _div(const double* a, const double* b, double* out){
	int i = threadIdx.x;
	out[i] = a[i]/b[i];
}

__global__ void _add(const double* a, const double b, double* out){
	int i = threadIdx.x;
	out[i] = a[i]+b;
}
__global__ void _sub(const double* a, const double b, double* out){
	int i = threadIdx.x;
	out[i] = a[i]-b;
}
__global__ void _mul(const double* a, const double b, double* out){
	int i = threadIdx.x;
	out[i] = a[i]*b;
}
__global__ void _div(const double* a, const double b, double* out){
	int i = threadIdx.x;
	out[i] = a[i]/b;
}

__global__ void _abs(const double* in, double* out){ //what if in == out? well...
	int i = threadIdx.x;
	out[i] = in[i]>0?in[i]:-in[i];
}

/* n >= 1024 */
__global__ void _add(const double* a, const double* b, double* out, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
		out[i] = a[i]+b[i];
}
__global__ void _sub(const double* a, const double* b, double* out, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		out[i] = a[i]-b[i];
}
__global__ void _mul(const double* a, const double* b, double* out, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		out[i] = a[i]*b[i];
}
__global__ void _div(const double* a, const double* b, double* out, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		out[i] = a[i]/b[i];
}

__global__ void _add(const double* a, const double b, double* out, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		out[i] = a[i]+b;
}
__global__ void _sub(const double* a, const double b, double* out, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		out[i] = a[i]-b;
}
__global__ void _mul(const double* a, const double b, double* out, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		out[i] = a[i]*b;
}
__global__ void _div(const double* a, const double b, double* out, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		out[i] = a[i]/b;
}


void add(const double* a, const double* b, double* o, int n){
	if(n < 1024){
		_add<<<1,n>>>(a,b,o);
	}else{
		int nb = (n + 255) / 256; //# of blocks
		_add<<<nb,256>>>(a,b,o,n);
	}
}
void sub(const double* a, const double* b, double* o, int n){
	if(n < 1024){
		_sub<<<1,n>>>(a,b,o);
	}else{
		int nb = (n + 255) / 256; //# of blocks
		_sub<<<nb,256>>>(a,b,o,n);
	}
}
void mul(const double* a, const double* b, double* o, int n){
	if(n < 1024){
		_mul<<<1,n>>>(a,b,o);
	}else{
		int nb = (n + 255) / 256; //# of blocks
		_mul<<<nb,256>>>(a,b,o,n);
	}
}
void div(const double* a, const double* b, double* o, int n){
	if(n < 1024){
		_div<<<1,n>>>(a,b,o);
	}else{
		int nb = (n + 255) / 256; //# of blocks
		_div<<<nb,256>>>(a,b,o,n);
	}
}

void add(const double* a, const double b, double* o, int n){
	if(n < 1024){
		_add<<<1,n>>>(a,b,o);
	}else{
		int nb = (n + 255) / 256; //# of blocks
		_add<<<nb,256>>>(a,b,o,n);
	}
}
void sub(const double* a, const double b, double* o, int n){
	if(n < 1024){
		_sub<<<1,n>>>(a,b,o);
	}else{
		int nb = (n + 255) / 256; //# of blocks
		_sub<<<nb,256>>>(a,b,o,n);
	}
}
void mul(const double* a, const double b, double* o, int n){
	if(n < 1024){
		_mul<<<1,n>>>(a,b,o);
	}else{
		int nb = (n + 255) / 256; //# of blocks
		_mul<<<nb,256>>>(a,b,o,n);
	}
}
void div(const double* a, const double b, double* o, int n){
	if(n < 1024){
		_div<<<1,n>>>(a,b,o);
	}else{
		int nb = (n + 255) / 256; //# of blocks
		_div<<<nb,256>>>(a,b,o,n);
	}
}

void abs(const double* in, double* out, int n){
	_abs<<<1,n>>>(in,out);
}
__global__ void _convolve(const double* d_i, const double* d_k, double* d_o,int r){
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
__global__ void _correlate(const double* d_i, const double* d_k, double* d_o,int r){
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
					* d_k[idx(r+ki,r+kj,2*r+1)]; //flipped here, for correlation
			}
			//effectively zero-padding
			//may change to VALID convolution later

			//d_o[i][j] += d_i[i+ki][j+kj] * d_k[ki+r][kj+r]
		}
	}
}
void convolve_d(const double* d_i, const double* d_k, double* d_o,
	//if all ptrs are in gpu
		int w, int h, int r,cudaStream_t* stream){
	dim3 g(1,1);
	dim3 b(w,h);
	if(stream){
		_convolve<<<g,b,0,*stream>>>(d_i,d_k,d_o,r);
	}else{
		_convolve<<<g,b>>>(d_i,d_k,d_o,r);
	}

}

void correlate_d(const double* d_i, const double* d_k, double* d_o,
	//if all ptrs are in gpu
		int w, int h, int r,cudaStream_t* stream){
	dim3 g(1,1);
	dim3 b(w,h);
	if(stream){
		_correlate<<<g,b,0,*stream>>>(d_i,d_k,d_o,r);
	}else{
		_correlate<<<g,b>>>(d_i,d_k,d_o,r);
	}
}

void convolve(const double* i, const double* k, double* o,
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
