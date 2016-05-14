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

	net.push_back(new ConvolutionLayer(12));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));
	net.push_back(new ConvolutionLayer(16));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));
	net.push_back(new FlattenLayer());
	net.push_back(new DenseLayer(84));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new DenseLayer(10));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new SoftMaxLayer());

	net.setup(Size(28,28), 1);
}


void train(ConvNet& net, int lim){

	std::ofstream ferr("error.csv");

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
		Yp[0].set_sync(false);
		namedPrint(Yp[0]);
		//namedPrint(Y[0]);

		net.BP(Yp,Y);
		ferr << net.error() << '\n';
	}

	ferr << std::endl;
	ferr.flush();
	ferr.close();

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
		namedPrint(Yp[0]);

		//namedPrint(Y[0]);
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
