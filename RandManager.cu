#include "RandManager.h"
#include <ctime>

RandManager::RandManager(){
    curandCreateGenerator (&rgen,CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed (rgen ,time(0));
}

RandManager::~RandManager(){

}

void RandManager::rand(double* arr, int n){
	//TODO : get rid of this.
	curandGenerateUniformDouble(rgen,arr,n);
	//curandGenerateNormalDouble(rgen,arr,n,0.0,0.1);
}
void RandManager::randu(double* arr, int n){
	curandGenerateUniformDouble(rgen,arr,n);
}

void RandManager::randn(double* arr, int n, double mean, double stddev){
	if(n%2 != 0){ //odd
		double* ptr;
		cudaMalloc(&ptr, (n+1)*sizeof(double));
		curandGenerateNormalDouble(rgen,ptr,(n+1),mean,stddev);
		cudaMemcpy(arr,ptr,n*sizeof(double),cudaMemcpyDeviceToDevice);
		cudaFree(ptr);
	}else{

		curandGenerateNormalDouble(rgen,arr,n,mean,stddev);
	}
}
