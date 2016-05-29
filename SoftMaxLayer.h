#ifndef __SOFTMAX_LAYER_H__
#define __SOFTMAX_LAYER_H__

#include "Layer.h"


class SoftMaxLayer: public Layer{
private:
	Size s;
	int d;
	std::vector<Matrix> O;
	std::vector<Matrix> G;
public:
	SoftMaxLayer();
	~SoftMaxLayer();
	virtual void setup(Size& s, int& d);

	virtual std::vector<Matrix>& FF(std::vector<Matrix>& I);
	virtual std::vector<Matrix>& BP(std::vector<Matrix>& G);
	virtual void update();

	double cost();

	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};

#endif
