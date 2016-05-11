#include "PoolLayer.h"

//TODO : research if custom data types can be passed through to GPU in CUDA


__global__ void pool(double* I, double* O, int* SW, //Switch
		int iw, //width of input matrix
		int s_w, int s_h,  //stride dims
		int p_w, int p_h){ //pool dims

	//it's max pooling.
	//is it in terms of magnitude?? or simply positive-max?
	//TODO: Research

	int h = blockDim.y;
	int w = blockDim.x;

	int i = threadIdx.y;
	int j = threadIdx.x;

	O[idx(i,j,w)] = -99999.0; // reasonably small value, anyways.

	for(int ii=0;ii<p_h;++ii){
		for(int jj=0;jj<p_w;++jj){
			int index = idx(i,j,w);
			double index_i = idx(s_h*i+ii,s_w*j+jj,iw);
			double val = I[index_i];
			if(val > O[index]){
				SW[index] = index_i; //switches, though stored in flattened index
				O[index] = val;
			}
		}
	}
}

__global__ void invert_pool(double* G_o, double* G_i, int* SW){

	int i = threadIdx.x;
	G_i[SW[i]] = G_o[i];

	//TODO : implement
}

PoolLayer::PoolLayer(Size s_s, Size s_p):s_s(s_s),s_p(s_p){

}

PoolLayer::~PoolLayer(){
	for(int i=0;i<d;++i){
		cudaFree(SW);
	}
}

void PoolLayer::setup(Size& s, int& d){
	s_in = s;
	this->d = d;

	int w = (s_in.w-s_p.w+s_s.w-1)/s_s.w;
	int h = (s_in.h-s_p.h+s_s.h-1)/s_s.h;
	s_out = Size(w,h);
	for(int i=0;i<d;++i){
		cudaMalloc(&SW[i],sizeof(int) * w*h);
	}

	s = s_out;
	//no change for d
}

std::vector<Matrix>& PoolLayer::FF(std::vector<Matrix>& _I){
	I = _I;

	dim3 blockDims(s_out.w, s_out.h);

	for(int i=0;i<d;++i){
		pool<<<1, blockDims>>>(I[i].d_data(),O[i].d_data(),SW[i],
				I[i].size().w,
				s_s.w, s_s.h,
				s_p.w, s_p.h
				);
	}
	return O;
}


std::vector<Matrix>& PoolLayer::BP(std::vector<Matrix>& _G){
	for(int i=0;i<d;++i){
		G[i].zero();
		invert_pool<<<1,s.wh>>>(_G[i],G[i],SW[i]);
	}
	return G;
}

void PoolLayer::update(){

}