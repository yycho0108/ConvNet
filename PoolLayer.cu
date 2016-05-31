#include "PoolLayer.h"

__global__ void pool(double* I, double* O, int* SW, //Switch
		int iw, int ih, //width of input matrix
		int s_w, int s_h,  //stride dims
		int p_w, int p_h){ //pool dims

	//TODO: is 'max' pooling in terms of magnitude? or positive-max only?

	int h = blockDim.y;
	int w = blockDim.x;

	int i = threadIdx.y;
	int j = threadIdx.x;

	double maxVal = -99999.0;// reasonably small value, anyways.
	int maxIdx = 0;

	int index = idx(i,j,w);
	//TODO : fix all these arbitrary numbers

	for(int ii=0;ii<p_h && s_h*i+ii < ih;++ii){ //check i+ii for bounds
		for(int jj=0;jj<p_w && s_w*j+jj < iw;++jj){ //check j+jj for bounds

			int index_i = idx(s_h*i+ii,s_w*j+jj,iw);
			double val = I[index_i];

			if(val > maxVal){
				maxIdx = index_i; //switches, stored in flattened index
				maxVal = val;
			}
		}
	}

	O[index] = maxVal;
	SW[index] = maxIdx;
}

__global__ void invert_pool(double* G_o, double* G_i, int* SW){

	int i = threadIdx.x;
	G_i[SW[i]] = G_o[i];
}

PoolLayer::PoolLayer(Size s_s, Size s_p):s_s(s_s),s_p(s_p){

}

PoolLayer::~PoolLayer(){
	for(int i=0;i<d;++i){
		cudaFree(SW[i]);
	}
	delete[] streams;
}

void PoolLayer::setup(Size& s, int& d){
	s_in = s;
	this->d = d;

	int w = s_in.w / s_s.w; //(s_in.w-s_p.w+s_s.w-1)/s_s.w;
	int h = s_in.h / s_s.h; //(s_in.h-s_p.h+s_s.h-1)/s_s.h;
	s_out = Size(w,h);

	SW.resize(d);

	streams = new cudaStream_t[d];
	for(int i=0;i<d;++i){
		cudaMalloc(&SW[i],sizeof(int) * w*h);
		//I.push_back(Matrix(s_in)); //doesn't need to allocate memory here
		G.push_back(Matrix(s_in));
		O.push_back(Matrix(s_out));
		cudaStreamCreate(&streams[i]);

	}

	s = s_out;
	//no change for d
}

std::vector<Matrix>& PoolLayer::FF(std::vector<Matrix>& _I){
	dim3 blockDims(s_out.w, s_out.h);

	for(int i=0;i<d;++i){
		//_I[i].copyTo(I[i]);
		pool<<<1, blockDims>>>(_I[i].d_data(),O[i].d_data(),SW[i],
				s_in.w, s_in.h,
				s_s.w, s_s.h,
				s_p.w, s_p.h
				);
	}
	return O;
}


std::vector<Matrix>& PoolLayer::BP(std::vector<Matrix>& _G){
	for(int i=0;i<d;++i){
		//_G[i].set_sync(false);
		//namedPrint(_G[i]);
		G[i].zero();
		invert_pool<<<1,s_out.wh>>>(_G[i].d_data(),G[i].d_data(),SW[i]);
		//G[i].set_sync(false);
		//namedPrint(G[i]);

	}
	return G;
}

void PoolLayer::update(){

}
