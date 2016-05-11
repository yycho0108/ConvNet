#ifndef __PARSER_H__
#define __PARSER_H__

#include <iostream>
#include <fstream>
#include <string>
#include "Matrix.h"

using namespace std;

class Parser{

private:
	ifstream f_d;
	ifstream f_l;
	unsigned char buf_d_raw[28*28];
	unsigned char buf_l_raw[1];

	double buf_d[28*28];
	double buf_l[10];

public:
	Parser(string d, string l);
	bool read(Matrix& d, Matrix& l);
	void reset();
	~Parser();
};

#endif
