# Simple Stochastic Optimizers Test
Simple stochastic optimizer comparison over a few primitive supervision.
The repository is aimed to test recent stochastic optimizers' basic capacity with a linear model,
and not any neural networks as we just want to see the convergence properties.
The implementation follows [scikit learn](https://scikit-learn.org/stable/) as
each algorithm has *fit* and *predict*. 

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

### requirement
* Python 3.4.0
* library  
	* matplotlib==2.0.1  
	* numpy==1.12.1  
	* scikit-learn==0.18.1
	* scipy==0.19.0

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
