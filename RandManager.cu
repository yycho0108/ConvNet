#include "RandManager.h"
#include <ctime>

__global__ void setup_rand( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void rand_gen(double* a, curandState* globalState)
{
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    a[ind] = RANDOM;
    globalState[ind] = localState;
}

RandManager::RandManager(int N):N(N){
	cudaMalloc(&s, N*sizeof(curandState));
	setup_rand<<<1,N>>>(s,time(0));
}

RandManager::~RandManager(){
	cudaFree(s);
}

void RandManager::rand(double* arr, int n){

	while (n > N) {
		rand_gen<<<1, N>>>(arr, s);
		arr += N;
		n -= N;
	}

	rand_gen<<<1,n>>>(arr,s);

}
