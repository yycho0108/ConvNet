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
#include "RandManager.h"

#include <iostream>
//#include <functional>


struct Matrix {
private:
	Size s;
	double* d_dat = nullptr; //gpu
	double* dat = nullptr; //cpu
	bool synced;
	static RandManager rnd;
public:
	Matrix(); //no setup at all
	Matrix(Size s, double* dat=nullptr);
	Matrix(int w, int h, double* dat=nullptr);
	Matrix(const Matrix&); //lvalue, copy data.
	Matrix(Matrix&&);//r-xvalue, steal data.
	~Matrix();

	Matrix& operator=(const Matrix&);
	Matrix& operator=(Matrix&&);

	Matrix& operator+=(const Matrix&);
	Matrix& operator-=(const Matrix&);
	Matrix& operator/=(const Matrix&);
	Matrix& operator%=(const Matrix&); //elem-wise mul

	Matrix& operator+=(const Matrix&&);
	Matrix& operator-=(const Matrix&&);
	Matrix& operator/=(const Matrix&&);
	Matrix& operator%=(const Matrix&&); //elem-wise mul

	Matrix& operator+=(double);
	Matrix& operator-=(double);
	Matrix& operator*=(double);
	Matrix& operator/=(double);

	Matrix operator+(const Matrix&) const;
	Matrix operator-(const Matrix&) const;
	Matrix operator*(const Matrix&) const; //dot product
	Matrix operator/(const Matrix&) const;
	Matrix operator%(const Matrix&) const; //elem-wise mul

	Matrix operator+(const Matrix&&) const;
	Matrix operator-(const Matrix&&) const;
	Matrix operator*(const Matrix&&) const; //dot product
	Matrix operator/(const Matrix&&) const;
	Matrix operator%(const Matrix&&) const; //elem-wise mul

	Matrix operator+(double) const;
	Matrix operator-(double) const;
	Matrix operator*(double) const;
	Matrix operator/(double) const;


	double operator()(int,int);

	Matrix& apply(double f(double)); //for each elem

	double max(Size* idx=nullptr); //max of all elem
	double min(Size* idx=nullptr); //min of all elem
	double sum(); //sum of all elem
	double avg(); //avg of all elem

	void zero(); //set to zero
	void one(); //set to zero
	void eye(); //set to identity
	void rand(); //set to rand
	void abs(); //set to positive

	void copyTo(Matrix& m) const; //copy to data, check for nullptr
	void transpose();

	static Matrix zeros(int w, int h);
	static Matrix zeros(Size s);
	static Matrix ones(int w, int h);
	static Matrix ones(Size s);
	static Matrix eye(int w, int h);
	static Matrix eye(Size s);
	static Matrix rand(int w, int h);
	static Matrix rand(Size s);

	static Matrix transpose(const Matrix&);
	static Matrix abs(const Matrix&);

	void sync(); //synchronizes device-host memory
	void sync_r(); //host to device
	void set_sync(bool); //set synchronization state
	void print(std::ostream& s); //visualizing Matrix Data

	//getters
	Size size() const;
	double* data() const; //cpu data
	double* d_data() const; //device data (gpu)
};

std::ostream& operator<<(std::ostream& os, Matrix& m);
bool isnan(Matrix& m);

#endif /* MATRIX_H_ */
