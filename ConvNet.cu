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

		for(auto& m : *I){

			m.set_sync(false);
			if(isnan(m)){
				namedPrint(m);
				throw "FF_NAN!";
			}
		}
		/* DEBUGGING END*/
		//namedPrint((*I)[0]);
		//take ptr only, no copy
	}
	return *I;
}

void ConvNet::BP(std::vector<Matrix>& O, std::vector<Matrix>& T){
	std::vector<Matrix> _G;
	//setup output gradient
	for(size_t i=0;i<O.size();++i){
		_G.push_back(T[i]-O[i]);
		//G[i] = Y[i] - Yp[i];
	}
	//hline();
	//namedPrint(_G[0]);
	//hline();
	auto G = &_G;
	for(auto i = L.rbegin()+1; i != L.rend(); ++i){
		auto& l = (*i);
		G = &l->BP(*G);

		for(auto& m : *G){
			m.set_sync(false);
			if(isnan(m)){
				throw "BP_NAN!";
			}
		}

		//namedPrint((*G)[0]);
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
