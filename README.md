##Convolutional Neural Network
---

Fully GPU-based Convolutional Neural Network Implementation

Features : 

- Custom Matrix Library
	- [x] Basic Arithmetic
		- Follows Armadillo's operator conventions
	- [x] Automatic Memory Management
	- [x] Utility functions
	- [x] Random-fill
	- [x] Shared Memory Optimization
	- [ ] __restrict__ Optimization
	- [ ] Save-Load Function
	- [x] Optimize for non in-place calculations by setting output ptr
	- [ ] Concurrent Execution Optimization
	- [ ] Increase Block Size to Hide Latency
	- [ ] Elliminate Bank Conflicts

#Layers :
- [x] Activation Layer
	- [x] ReLU
	- [x] Sigmoid
	- [x] Softplus
	- [x] Tanh
	- [ ] Linear
- [x] Convolution Layer
	- [x] *SAME* Convolution
	- [ ] *VALID* Convolution
	- [ ] *FULL* Convolution
- [x] Dense Layer
- [x] Dropout Layer
- [x] Flatten Layer
- [x] Pool Layer
- [x] Output Layer
	- [x] SoftMax
	- [ ] Linear


# TASKS

- [ ] Adaptive Learning Rate Optimizations
	- [ ] AdaDelta
	- [ ] AdaGrad
	- [ ] Adam
	- [ ] RMSProp
- [x] Momentum
	- [ ] Nesterov Momentum
- [x] MiniBatch Gradient Descent
	- [ ] Parallelize MiniBatch Learning
- [x] Stochastic Gradient Descent
- [ ] Conv-Pool Layer for Speed
- [ ] Detecting End of Training
- [x] Reimplement *DeConvolution* to work faster

# RESULTS

90% Accuracy on MNIST Data

![Loss](images/error.png)

Training Loss. The network improves slowly ever since ~12000th iterations.

![Result](images/result.png)

Final Collected Result
