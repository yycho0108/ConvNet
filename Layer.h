/*
 * Layer.h
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#ifndef __LAYER_H_
#define __LAYER_H_

#include <vector>

#include "Size.h"
#include "Utility.h"
#include "Matrix.h"

class Layer {
public:
	//Layer(); no need for constructor
	virtual ~Layer();

	virtual std::vector<Matrix>& FF(std::vector<Matrix>&)=0;
	virtual std::vector<Matrix>& BP(std::vector<Matrix>&)=0;

	virtual void update(){};
	virtual void setup(Size&,int&){};//int for "depth" of previous.

	virtual void save(FileStorage& f, int i)=0;
	virtual void load(FileStorage& f, int i)=0;
};

#endif /* LAYER_H_ */
