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
	//curandGenerateNormalDouble(rgen,arr,n,0.0,1.0);
}
void RandManager::randu(double* arr, int n){
	curandGenerateUniformDouble(rgen,arr,n);
}

void RandManager::randn(double* arr, int n, double mean=0.0, double stddev=1.0){
	curandGenerateNormalDouble(rgen,arr,n,mean,stddev);
}
