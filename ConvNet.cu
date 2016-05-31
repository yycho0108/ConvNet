#include "ConvNet.h"

//RMS error
double RMS(Matrix& m){
	return sqrt((m%m).avg());
	//square -> mean -> root
}

ConvNet::ConvNet(){

}

ConvNet::~ConvNet(){
	for(auto& l : L){
		delete l;
	}
}

std::vector<Matrix>& ConvNet::FF(std::vector<Matrix>& _I){
	auto I = &_I; //ptr to vector
	for(auto& l : L){
		I = &(l->FF(*I));
		/* DEBUGGING START*/

		/*for(auto& m : *I){

			m.set_sync(false);
			if(isnan(m)){
				namedPrint(m);
				throw "FF_NAN!";
			}
		}*/
		/* DEBUGGING END*/
		//(*I)[0].set_sync(false);
		//namedPrint((*I)[0]);
		//take ptr only, no copy
	}
	return *I;
}

struct ff_info{
	int id;
	int b_id;
	std::vector<Layer*> *L;
	Batch_t *I;
	Batch_t *T;
};

void* FFBP_wrap(void* args){
	ff_info* info = (ff_info*) args;

	int id = info->id;
	int b_id = info->b_id; //batch id
	int nL = info->L->size();

	std::vector<Layer*> *L = info->L;
	std::vector<Matrix> *I = &info->I[b_id];
	std::vector<Matrix> *T = &info->T[b_id];

	// FF...
	for(int l=0;l<nL;++l){
		//serial op.
		I = &(*L)[l]->FF(*I, id);
		//id tells the layer where to store the input.
		//other than that, exactly the same.
	}

	std::vector<Matrix> _G = *T - *I; // I == O in this case

	auto G = &_G;

	for(auto l = nL-2; l >= 0; --l){
		G = &(*L)[l]->BP(*G,id);
	}

	pthread_exit(NULL);
}

void ConvNet::FFBP(Batch_t& _I, Batch_t& _T, std::vector<int>& indices){
	int n = indices.size();
	pthread_t* threads = new pthread_t[n];
	ff_info* info = new ff_info[n];

	for(int i=0;i<n;++i){
		info[i] = {i,indices[i],&L,&_I};
		pthread_create(&threads[i],nullptr,FFBP_wrap,(void*) &info[i]);
	}

	for(int i=0;i<n;++i){
		pthread_join(threads[i],NULL);
	}
	delete[] threads;
	delete[] ff_info;
}
void ConvNet::BP(std::vector<Matrix>& O, std::vector<Matrix>& T){
	std::vector<Matrix> _G;
	//setup output gradient
	for(size_t i=0;i<O.size();++i){
		_G.push_back(T[i]-O[i]);
		//G[i] = Y[i] - Yp[i];
	}
	loss = RMS(_G[0]);

	//hline();
	//namedPrint(_G[0]);
	//hline();
	auto G = &_G;
	for(auto i = L.rbegin()+1; i != L.rend(); ++i){
		auto& l = (*i);
		G = &l->BP(*G);

		/*for(auto& m : *G){
			m.set_sync(false);
			if(isnan(m)){
				throw "BP_NAN!";
			}
		}*/

		//namedPrint((*G)[0]);
	}

}


void ConvNet::push_back(Layer*&& l){
	L.push_back(l);
	//take ownership
	l = nullptr;
}

void ConvNet::setup(Size s, int d){
	for(auto& l : L){
		l->setup(s,d);
	}
}
void ConvNet::update(){
	for(auto& l : L){
		l->update();
	}
}
double ConvNet::error(){
	return loss;
}

void ConvNet::debug(){
	L[0]->debug();
}
