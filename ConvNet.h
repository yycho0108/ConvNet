#ifndef __CONVNET_H__
#define __CONVNET_H__

#include "Layer.h"
#include "ActivationLayer.h"
#include "ConvolutionLayer.h"
#include "DenseLayer.h"
#include "DropoutLayer.h"
#include "FlattenLayer.h"
#include "PoolLayer.h"
#include "SoftMaxLayer.h"
#include <vector>

class ConvNet{

private:
	std::vector<Layer*> L; //layers. set as ptr, to avoid copying-syntax when pushing
	double loss; //most recent loss
public:
	ConvNet();
	~ConvNet();

	std::vector<Matrix>& FF(std::vector<Matrix>& _I);
	void FFBP(Batch_t& _I, Batch_t& _T, std::vector<int>& indices);
	void BP(std::vector<Matrix>& O, std::vector<Matrix>& T);

	void setup(Size s, int d); //size & depth of input
	void push_back(Layer*&& l);
	void update();
	double error();
	void debug();
	//void save(std::string dir);//save directory
	//void load(std::string dir);
};
#endif
