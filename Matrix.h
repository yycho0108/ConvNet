/*
 * Matrix.h
 *
 *  Created on: May 7, 2016
 *      Author: jamiecho
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "Size.h"
#include "Utility.h"
#include <functional>


struct Matrix {
private:
	Size s;
	double* d_dat = nullptr; //gpu
	double* dat = nullptr; //cpu
	Matrix* t; //transpose.
public:
	Matrix(); //no setup at all
	Matrix(Size s, double* dat=nullptr);
	Matrix(int n, int m, double* dat=nullptr);
	Matrix(Matrix&); //lvalue, copy data.
	Matrix(Matrix&&);//r-xvalue, steal data.
	~Matrix();

	Matrix& operator+=(Matrix&);
	Matrix& operator-=(Matrix&);
	Matrix& operator*=(Matrix&); //dot product
	Matrix& operator/=(Matrix&);
	Matrix& operator%=(Matrix&); //elem-wise mul

	Matrix& operator+=(Matrix&&); //rvalues
	Matrix& operator-=(Matrix&&);
	Matrix& operator*=(Matrix&&); //dot product
	Matrix& operator/=(Matrix&&);
	Matrix& operator%=(Matrix&&); //elem-wise mul

	Matrix& operator+=(double);
	Matrix& operator-=(double);
	Matrix& operator*=(double);
	Matrix& operator/=(double);

	Matrix operator+(Matrix&);
	Matrix operator-(Matrix&);
	Matrix operator*(Matrix&); //dot product
	Matrix operator/(Matrix&);
	Matrix operator%(Matrix&); //elem-wise mul

	Matrix operator+(Matrix&&);
	Matrix operator-(Matrix&&);
	Matrix operator*(Matrix&&); //dot product
	Matrix operator/(Matrix&&);
	Matrix operator%(Matrix&&); //elem-wise mul

	Matrix operator+(double);
	Matrix operator-(double);
	Matrix operator*(double);
	Matrix operator/(double);

	Matrix& operator=(Matrix& m);
	Matrix& operator=(Matrix&& m);

	std::vector<double>& operator()(int,int);

	Matrix& T(); //transpose

	Matrix& apply(double f(double)); //for each elem

	double max(); //max of all elem
	double min(); //min of all elem
	double sum(); //sum of all elem
	double avg(); //avg of all elem
	void zero(); //set to zero

	void copyTo(Matrix& m); //copy to data, check for nullptr

	static Matrix eye(int n, int m);
	static Matrix eye(Size s);
	static Matrix zeros(int n, int m);
	static Matrix zeros(Size s);
	static Matrix ones(int n, int m);
	static Matrix ones(Size s);
	static Matrix rand(int n, int m);
	static Matrix rand(Size s);
	static Matrix empty(int n, int m);
	static Matrix empty(Size s);

	static Matrix transpose(Matrix&);


	void sync(); //synchronizes device-host memory

	//getters
	Size size();
	double* data(); //cpu data
	double* d_data(); //device data (gpu)
};

#endif /* MATRIX_H_ */
