#ifndef __POOL_LAYER_H__
#define __POOL_LAYER_H__

#include "Layer.h"

class PoolLayer : public Layer{
	Size s_in, s_out;
	Size s_s,s_p; //pooling size, stride size

	int d;
	std::vector<std::vector<std::vector<Size>>> S; //switches

	std::vector<int*> SW; //Switch list
	std::vector<Matrix> I;
	std::vector<Matrix> O;
	std::vector<Matrix> G;

public:
	PoolLayer(Size s_s, Size s_p);
	virtual void setup(Size& s, int& d);

	virtual std::vector<Matrix>& FF(std::vector<Matrix>& I);
	virtual std::vector<Matrix>& BP(std::vector<Matrix>& G);
	virtual void update();

	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};

#endif