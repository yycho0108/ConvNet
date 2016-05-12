#include "RandManager.h"
#include <ctime>

RandManager::RandManager(){
    curandCreateGenerator (&rgen,CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed (rgen ,time(0));
}

RandManager::~RandManager(){

}

void RandManager::rand(double* arr, int n){
	curandGenerateUniformDouble(rgen,arr,n);
}
