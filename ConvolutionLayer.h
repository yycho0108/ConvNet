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
	double m; //momentum
	std::vector<Matrix> W, B, dW, dW_p, dB, dB_p, I, O, G;

public:
	ConvolutionLayer(int d_out=1); //# kernels
	virtual ~ConvolutionLayer();

	virtual std::vector<Matrix>& FF(std::vector<Matrix>& I);
	virtual std::vector<Matrix>& BP(std::vector<Matrix>& G);

	virtual void update();

	virtual void setup(Size&,int&);
	std::vector<Matrix>& getW();

};

#endif /* CONVOLUTIONLAYER_H_ */
