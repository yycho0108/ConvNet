#include "Parser.h"

Parser::Parser(string d, string l){
	f_d.open(d);
	f_l.open(l);

	if(f_d.fail() || f_l.fail()){
		std::cerr << "ERROR : failed to open [" << d << ',' << l << "]:" << strerror(errno) << std::endl;
	}
	reset();
}

bool Parser::read(Matrix& d, Matrix& l){
	//std::cout << "READING " << std::endl;

	f_d.read((char*)buf_d_raw,28*28);
	f_l.read((char*)buf_l_raw,1);

	//get img to Matrix
	for(int i=0;i<28;++i){
		for(int j=0;j<28;++j){
			auto index = idx(i,j,28);
			buf_d[index] = buf_d_raw[index] / 256.0; //normalize
		}
	}

	//get label to Matrix
	memset(buf_l,0,10*sizeof(double));
	buf_l[buf_l_raw[0]] = 1.0;

	d = Matrix(28,28,buf_d);
	l = Matrix(1,10,buf_l);

	return f_d && f_l;
}

void Parser::reset(){
	f_d.clear();
	f_d.seekg(16,ios::beg);
	f_l.clear();
	f_l.seekg(8,ios::beg);
}

Parser::~Parser(){
	f_d.close();
	f_l.close();
}

