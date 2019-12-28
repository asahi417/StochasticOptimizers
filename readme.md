# Simple Stochastic Optimizers Test
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)

Simple stochastic optimizer comparison over classification/regression tasks.
The repository is aimed to test recent stochastic optimizers' basic capacity with a linear model,
and not any neural networks as we just want to see the convergence properties.
The implementation follows [scikit learn](https://scikit-learn.org/stable/) as
each algorithm has *fit* and *predict*. 

## Get started
```
git clone https://github.com/asahi417/StochasticOptimizers
cd StochasticOptimizers
pip install .
```

* learning curve
	* sklearn's learning curve comparing split data
	* This library use warm_start (save time)
* multilabel classification
	* skleran's OneVsRestClassifier cant use warm_start
	* This library include OVR classifier base which can use warm_start
* plot curves
	* learning curve with 95% CI
* logger
	* learning progress can be seen in logger file

### estimator
* SGD (Stochastic Gradient Descent)
* FOBOS (Forward Backward Splitting)
* APFBS (Adaptive Proximal Forward Backward Splitting)
* SDA (Stochastic Dual Averaging)
* RDA (Reguralized Dual Averaging)
* PDA (Projection-based Dual Averaging)
* AdaGrad (RDA type & FOBOS type)
* Variance SGD 
* AdaDelta
* Adam
* RMSprop

* building....
	* TONGA, SGD-QN, sLBFGS

### framework
* Testing Framework (sklearn based)
	* GridSearch
		* grid search method
	* LearningCurve
		* compare different online estimator
	* OneVsRestClassifier

### Other
* PlotCurves
	* plot learning curves

### usage
* some example is set in "example/"
* almost as same as sklearn
