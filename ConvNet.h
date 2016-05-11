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
	ConvNet();
	~ConvNet();

	std::vector<Layer*> L; //layers. set as ptr, to avoid copying-syntax when pushing
	std::vector<Matrix>& FF(std::vector<Matrix>& _X);
	void BP(std::vector<Matrix>& O, std::vector<Matrix>& T);

	void setup(Size s, int d); //size & depth of input
	void push_back(Layer*&& l);

	//void save(std::string dir);//save directory
	//void load(std::string dir);
};
#endif