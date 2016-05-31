#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

#define NUM_THREADS 10000

#define N 50
#define M 50

#define L 4

#define idx(i,j,w) ((i)*(w)+(j))

template<typename T>
void print2d(T* arr, int h, int w){
	for(int i=0;i<h;++i){
		for(int j=0;j<w;++j){
			std::cout << arr[idx(i,j,w)] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << std::endl;
}


class Layer{
	int* i_data;
	public:
	void ff(int* i_data, int* o_data, int i){
		memcpy(this->i_data, i_data, sizeof(int) * M); // storage
		for(int j=0;j<M;++j){
			o_data[idx(i,j,M)] = i_data[idx(i,j,M)] + 1;
		}
	}
};

struct ff_info{
	int i; // index
	int nL; //# layer
	Layer* l;
	int* i_data;
	int* o_data;
};

void* ff_wrap(void* args){
	ff_info* info = (ff_info*) args;

	//unpack
	int i = info->i; // 0~N
	int nL = info->nL;
	Layer* l = info->l;
	int* i_data = info->i_data;
	int* o_data = info->o_data;

	for(int it=0;it<nL;++it){
		l[it].ff(i_data, o_data, i);
		/* ** copy o to i ** */
		memcpy(&i_data[i*M],&o_data[i*M],sizeof(int)*M);
	}

	pthread_exit(NULL);
}
class Net{

private:
	Layer l[L];

public:
	void ff(int i_data[N][M], int o_data[N][M]){
		pthread_t threads[NUM_THREADS];
		ff_info info[NUM_THREADS];
		//.. launch N threads
		for(int i=0; i<N; ++i){
			info[i] = {i,L, l,(int*)i_data,(int*)o_data};
			pthread_create(&threads[i], nullptr, ff_wrap, (void*) &info[i]);
		}

		for(int i=0;i<N;++i){
			pthread_join(threads[i],NULL);
		}
	}

};
int main(){
	Net net;

	int i_data[N][M];

	for(int i=0;i<N;++i){
		for(int j=0;j<M;++j){
			i_data[i][j] = i+j;
		}
	}

	int o_data[N][M];

	print2d((int*)i_data,N,M);

	net.ff(i_data,o_data);

	print2d((int*)o_data,N,M);

	pthread_exit(NULL);
}
