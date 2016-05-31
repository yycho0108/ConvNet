/*
 * ConvolutionLayer.h
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#ifndef __CONVOLUTION_LAYER_H__
#define __CONVOLUTION_LAYER_H__


#include "Layer.h"
#include "Matrix.h"

#include <vector>

class ConvolutionLayer: public Layer {
private:
	Size s;
	int d_in, d_out;
	bool** connection;
	std::vector<Matrix> W, B, dW, dB, I, O, G;
	std::vector<Matrix> dW_p, dB_p;
	std::vector<Matrix> dW_t, dB_t; //-> mini-batch
	cudaStream_t* streams_i;
	cudaStream_t* streams_o;

public:
	ConvolutionLayer(int d_out=1); //# kernels
	~ConvolutionLayer();
	virtual void setup(Size&,int&);

	virtual std::vector<Matrix>& FF(std::vector<Matrix>& I);
	virtual std::vector<Matrix>& BP(std::vector<Matrix>& G);
	virtual void update();
	virtual void debug();
};

#endif /* CONVOLUTIONLAYER_H_ */
