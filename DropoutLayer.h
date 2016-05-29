#ifndef __DROPOUT_LAYER_H__
#define __DROPOUT_LAYER_H__

//TODO : Implement

#include "Layer.h"


class DropoutLayer : public Layer{
private:
	int d;
	double p; //dropout probability
	Size s;
	static bool enabled;
	cudaStream_t* streams;
	std::vector<Matrix> O;
	std::vector<Matrix> Mask;
public:
	DropoutLayer(double p=0.5);
	~DropoutLayer();
	virtual void setup(Size& s, int& d);

	virtual std::vector<Matrix>& FF(std::vector<Matrix>& I);
	virtual std::vector<Matrix>& BP(std::vector<Matrix>& G);
	virtual void update();
	static void enable(bool);
	//no need to update since to trainable parameter
	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};
#endif
