
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda.h>

#include "ConvNet.h"
#include "Parser.h"


static volatile bool keepTraining = true;
static volatile bool keepTesting = true;

void intHandler(int){
	if(keepTraining){
		keepTraining = false;
	}else{
		keepTesting = false;
	}
}


void setup(ConvNet& net){
	/* ** CONV LAYER TEST ** */

	/*net.push_back(new ConvolutionLayer(12));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));
	net.push_back(new ConvolutionLayer(16));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));
	*/
	net.push_back(new FlattenLayer());
	net.push_back(new DenseLayer(84));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new DenseLayer(10));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new SoftMaxLayer());

	net.setup(Size(28,28), 1);
}


void train(ConvNet& net, int lim){
	std::cout << "TRAINING FOR : " << lim << std::endl;
	keepTraining = true;

	Parser trainer("data/trainData","data/trainLabel");
	std::vector<Matrix> X(1),Y(1);

	int i = 0;

	while (1){
		if(!trainer.read(X[0],Y[0])){//EOF
			trainer.reset();
		}

		//X[0].print(std::cout);

		if(++i > lim || !keepTraining)
			return;

		//if(!(i%100)){
			cout << "TRAINING ... " << i << endl;
		//}
		auto& Yp = net.FF(X);
		//namedPrint(Yp[0]);
		//namedPrint(Y[0]);

		net.BP(Yp,Y);
	}

	keepTraining = false;
}

void test(ConvNet& net){
	keepTesting = true;

	Parser tester("data/testData","data/testLabel");

	Matrix d,l;
	std::vector<Matrix> X(1),Y(1);

	int cor = 0;
	int inc = 0;

	while(tester.read(X[0],Y[0]) && keepTesting){ //read into X,Y
		//namedPrint(X[0]);

		/* VISUALIZATION START */
		/*auto i_ptr = X[0].data();
		for(int i=0;i<28;++i){
			for(int j=0;j<28;++j){
				std::cout << (i_ptr[idx(i,j,28)]>0.5?'1':'0');
			}
			std::cout << '\n';
		}
		std::cout << std::endl;*/
		/* VISUALIZATION END */

		Size y;
		Size t;
		auto& Yp = net.FF(X);
		Yp[0].set_sync(false); //--> to force sync
		//namedPrint(Y[0]);
		//namedPrint(Yp[0]);
		Yp[0].max(&y);
		Y[0].max(&t);
		//namedPrint(y);
		//namedPrint(t);

		(y==t)?(++cor):(++inc);
		cout << "y[" << y.h << "]:T[" << t.h <<"]"<<endl;

		printf("%d cor, %d inc\n", cor,inc);
	}

	keepTesting = false;
}

int main(int argc, char* argv[]){

	/*
	std::ofstream f_check("check.txt"); //checking working directory
	f_check << "CHECK " << std::endl;
	f_check.flush();
	f_check.close();
	*/
	int lim = 60000;

	if(argc > 1){
		lim = std::atoi(argv[1]);
	}

	ConvNet net;
	setup(net);
	train(net, lim);
	test(net);
}
//
//static const int WORK_SIZE = 256;
//
///**
// * This macro checks return value of the CUDA runtime call and exits
// * the application if the call failed.
// *
// * See cuda.h for error code descriptions.
// */
//#define CHECK_CUDA_RESULT(N) {											\
//	CUresult result = N;												\
//	if (result != 0) {													\
//		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
//			result);													\
//		exit(1);														\
//	} }
//
//int main(int argc, char **argv)
//{
//	CUmodule module;
//	CUcontext context;
//	CUdevice device;
//	CUdeviceptr deviceArray;
//	CUfunction process;
//
//	void *kernelArguments[] = { &deviceArray };
//	int deviceCount;
//	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];
//
//	for (int i = 0; i < WORK_SIZE; ++i) {
//		idata[i] = i;
//	}
//
//	CHECK_CUDA_RESULT(cuInit(0));
//	CHECK_CUDA_RESULT(cuDeviceGetCount(&deviceCount));
//	if (deviceCount == 0) {
//		printf("No CUDA-compatible devices found\n");
//		exit(1);
//	}
//	CHECK_CUDA_RESULT(cuDeviceGet(&device, 0));
//	CHECK_CUDA_RESULT(cuCtxCreate(&context, 0, device));
//
//	CHECK_CUDA_RESULT(cuModuleLoad(&module, "bitreverse.fatbin"));
//	CHECK_CUDA_RESULT(cuModuleGetFunction(&process, module, "bitreverse"));
//
//	CHECK_CUDA_RESULT(cuMemAlloc(&deviceArray, sizeof(int) * WORK_SIZE));
//	CHECK_CUDA_RESULT(
//			cuMemcpyHtoD(deviceArray, idata, sizeof(int) * WORK_SIZE));
//
//	CHECK_CUDA_RESULT(
//			cuLaunchKernel(process, 1, 1, 1, WORK_SIZE, 1, 1, 0, NULL, kernelArguments, NULL));
//
//	CHECK_CUDA_RESULT(
//			cuMemcpyDtoH(odata, deviceArray, sizeof(int) * WORK_SIZE));
//
//	for (int i = 0; i < WORK_SIZE; ++i) {
//		printf("Input value: %u, output value: %u\n", idata[i], odata[i]);
//	}
//
//	CHECK_CUDA_RESULT(cuMemFree(deviceArray));
//	CHECK_CUDA_RESULT(cuCtxDestroy(context));
//
//	return 0;
//}
