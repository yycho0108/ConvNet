#ifndef __FLATTEN_LAYER_H__
#define __FLATTEN_LAYER_H__


#include "Layer.h"

class FlattenLayer : public Layer {
private:
	int d_in;
	Size s_in,s_out;
	cudaStream_t* streams;

	std::vector<Matrix> O;
	std::vector<Matrix> G;
public:
	FlattenLayer();
	~FlattenLayer();
	virtual void setup(Size&,int&);//int for "depth" of previous.

	virtual std::vector<Matrix>& FF(std::vector<Matrix>&);
	virtual std::vector<Matrix>& BP(std::vector<Matrix>&);
	virtual void update();

	//virtual void save(FileStorage& f, int i)=0;
	//virtual void load(FileStorage& f, int i)=0;
};

#endif
