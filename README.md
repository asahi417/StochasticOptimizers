# Simple Stochastic Optimizers Test

Simple stochastic optimizer comparison over classification/regression tasks.
The repository was originally designed as a supplemental material for
my lecture for stochastic optimization ([slide](https://asahi417.github.io/assets/slides/stochastic_optimization_slide.pdf)),   
which was aimed to test recent stochastic optimizers' basic capacity with a linear model,
and not any neural networks as we just want to see the convergence properties.
The implementation follows [scikit learn](https://scikit-learn.org/stable/) as
each algorithm has *fit* and *predict*. 
One can find sample use case in [notebook](https://github.com/asahi417/StochasticOptimizers/blob/master/example/classification_mnist.ipynb),
and figure 1 is the learning curve from it.

<p align="center">
  <img src="./example/example_results/linear_model_mnist/error.png" width="500">
  <br><i>Fig 1: Learning curve result on MNIST </i>
</p>


## Get started
```
git clone https://github.com/asahi417/StochasticOptimizers
cd StochasticOptimizers
pip install .
```

## Optimizers
- ***Gradient Descents***
    - stochastic gradient descent (nesterov's acceleration and momentum), [code](./stochastic_optimizer/estimator/SGD.py) 
    - Forward Backward Splitting (FOBOS), [code](./stochastic_optimizer/estimator/FOBOS.py), [paper](http://www.jmlr.org/papers/volume10/duchi09a/duchi09a.pdf)
    - Adaptive Proximal Forward Backward Splitting (APFBS), [code](./stochastic_optimizer/estimator/APFBS.py), [paper](https://www.arl.nus.edu.sg/twiki6/pub/ARL/BibEntries/Pelekanakis_and_Chitre_2014_Adaptive_Sparse_Channel.pdf)
    - AdaGrad, [code](./stochastic_optimizer/estimator/AdaGrad.py), [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    - Variance SGD, [code](./stochastic_optimizer/estimator/VSGD.py), [paper](https://arxiv.org/pdf/1206.1106.pdf)
    - AdaDelta, [code](./stochastic_optimizer/estimator/AdaDelta.py), [paper](https://arxiv.org/pdf/1212.5701.pdf)
    - Adam, [code](./stochastic_optimizer/estimator/Adam.py), [paper](https://arxiv.org/pdf/1412.6980.pdf%20%22%20entire%20document)
    - RMSprop, [code](./stochastic_optimizer/estimator/rmsprop.py), [slide](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- ***Dual Averaging***
    - stochastic dual averaging (SDA), [code](./stochastic_optimizer/estimator/SDA.py), [paper](http://ium.mccme.ru/postscript/s12/GS-Nesterov%20Primal-dual.pdf)
    - reguralized dual averaging (RDA), [code](./stochastic_optimizer/estimator/RDA.py), [paper](http://www.jmlr.org/papers/volume11/xiao10a/xiao10a.pdf)
    - projection-based dual averaging (PDA), [code](./stochastic_optimizer/estimator/PDA.py), [paper](https://asahi417.github.io/assets/papers/tsp_pda_with_bio.pdf), [slide](https://asahi417.github.io/assets/slides/icassp_17_asahi.pdf)
