#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__

#include "Layer.h"
#include <string>

typedef double (*dfun)(double);

class ActivationLayer : public Layer{
private:
	int d;
	Size s;
	dfun f;
	dfun f_d;
	cudaStream_t* streams;
	std::vector<Matrix>* pI;
	std::vector<Matrix> O;
public:
	ActivationLayer(std::string _f);
	~ActivationLayer();
	virtual void setup(Size& s, int& d);

	virtual std::vector<Matrix>& FF(std::vector<Matrix>& I);
	virtual std::vector<Matrix>& BP(std::vector<Matrix>& G);
	virtual void update();
	//no need to update since to trainable parameter
	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};
#endif
