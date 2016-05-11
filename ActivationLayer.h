#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__

#include "Layer.h"
#include <string>

class ActivationLayer : public Layer{
private:
	int d;
	Size s;

	double (*f)(double); //device functions.
	double (*f_d)(double);

	std::vector<Matrix> I;
	std::vector<Matrix> O;
	std::vector<Matrix> G; //maybe not necessary? idk...
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
