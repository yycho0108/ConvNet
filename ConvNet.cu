#include "ConvNet.h"


ConvNet::ConvNet(){

}

ConvNet::~ConvNet(){
	for(auto& l : L){
		delete l;
	}
}

std::vector<Matrix>& ConvNet::FF(std::vector<Matrix>& _I){
	auto& I = _I;
	for(auto& l : L){
		I = l->FF(I);
	}
	return I;
}

void ConvNet::BP(std::vector<Matrix>& O, std::vector<Matrix>& T){
	std::vector<Matrix> G;
	for(size_t i=0;i<G.size();++i){
		G.push_back(Y[i]-Yp[i]);
		//G[i] = Y[i] - Yp[i];
	}

	for(auto i = L.rbegin()+1; i != L.rend(); ++i){
		auto& l = (*i);
		G = l->BP(G);
	}

	for(auto& l : L){
		l->update();
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
