/*
 * Utility.h
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#define idx(i,j,w) ((i)*(w)+(j))
#define inbound(i,j,n,m) (0<=(i) && 0<=(j) && (i)<(n) && (j)<(m))
#define LOOP(start,end,content) \
	for(int _it = start; _it < end; ++_it){ \
		content; \
	} \

#define namedPrint(x) \
	std::cout << #x << " : " << x << std::endl;


extern void convolve_d(double* i, double* k, double* o,
		int n, int m, int r);

extern void convolve(double* i, double* k, double* o,
		int n, int m, int r);

extern void correlate_d(double* d_i, double* d_k, double* d_o,
		int n, int m, int r);

extern void add(double* a, double* b, double* o, int n);
extern void sub(double* a, double* b, double* o, int n);
extern void mul(double* a, double* b, double* o, int n);
extern void div(double* a, double* b, double* o, int n);
extern void add(double* a, double b, double* o, int n);
extern void sub(double* a, double b, double* o, int n);
extern void mul(double* a, double b, double* o, int n);
extern void div(double* a, double b, double* o, int n);

#endif /* UTILITY_H_ */
