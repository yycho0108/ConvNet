#ifndef __DENSE_LAYER_H__
#define __DENSE_LAYER_H__

#include "Layer.h"

class DenseLayer : public Layer{
private:
	int s_i,s_o; //no depth. just no.
	cudaStream_t stream;
	Matrix W, dW,
		   B, dB,
		   dW_t, dB_t,
		   dW_p, dB_p;

	std::vector<Matrix> I, O, G;
public:
	DenseLayer(int s_out); //and possibly also optimization as arg.
	~DenseLayer();
	virtual void setup(Size& s, int& d);

	virtual std::vector<Matrix>& FF(std::vector<Matrix>& I);
	virtual std::vector<Matrix>& BP(std::vector<Matrix>& G);
	virtual void update();

	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};

#endif
