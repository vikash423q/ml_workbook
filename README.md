
<h1 align="center">ML Workbook</h1>

## Description
These are my initial footsteps into Machine Learning.
I have tried to implement ML/DL algorithms from scratch using numpy, because getting your hands dirty is the best way to learn.


## Comparision of optimization Algorithms on MNIST dataset
[Source code](./src/dl/example.py)

I have compared the performance of a Dense network with different optimizers.
Models used same test and train datasets. Here are the results.
​
<p align="justify"> 
    <img width="400" src="./temp/plot/nn/mnist/gd/accuracy.png" alt="Gradient Descent">
</p>

<p align="justify"> 
    <img width="400" src="./temp/plot/nn/mnist/gd/loss.png" alt="Gradient Descent">
</p>
​
<p align="justify"> 
    <img width="400" src="./temp/plot/nn/mnist/momentum/accuracy.png" alt="Momentum Optimizer">
</p>

<p align="justify"> 
    <img width="400" src="./temp/plot/nn/mnist/momentum/loss.png" alt="Momentum Optimizer">
</p>

<p align="justify"> 
    <img width="400" src="./temp/plot/nn/rmsprop/accuracy.png" alt="RMSProp Optimizer">
</p>

<p align="justify"> 
    <img width="400" src="./temp/plot/nn/rmsprop/loss.png" alt="RMSProp Optimizer">
</p>

<p align="justify"> 
    <img width="400" src="./temp/plot/nn/adam/accuracy.png" alt="Adam Optimizer">


<p align="justify"> 
    <img width="400" src="./temp/plot/nn/adam/loss.png" alt="Adam Optimizer">
</p>

## Comparision of some common ML classification Algorithms 
[Source code](./src/misc/classifier_comparision.py)

I have tried to compare the performance of some common ML Algorithms on sklearn datasets.
I have also compared Sklearn's implementation with my scratch implementation with side by side.

Models used same test and train datasets. Here are the results.
​
<p align="center"> 
    <img width="500" src="./temp/plot/ml/iris_accuracy.png" alt="iris accuracy">
</p>
​
<p align="center"> 
    <img width="500" src="./temp/plot/ml/wine_accuracy.png" alt="wine accuracy">
</p>

<p align="center"> 
    <img width="500" src="./temp/plot/ml/breast_cancer_accuracy.png" alt="breast cancer accuracy">
</p>
​

## Clustering data with K-Means
[Source code](./src/ml/k_means/example.py)
​
<p align="center"> 
    <img width="500" src="./temp/plot/kmeans/plot_scratch.png" alt="Kmeans-iris-scratch">
</p>
​
<p align="center"> 
    <img width="500" src="./temp/plot/kmeans/plot_sklearn.png" alt="Kmeans-iris-sklearn">
</p>

<p align="center"> 
    <img width="500" src="./temp/plot/kmeans/elbow.png" alt="Kmeans-iris-elbow">
</p>


## Dimensionality Reduction with PCA
[Source code](./src/ml/pca/example.py)
​
<p align="center"> 
    <img width="500" src="./temp/plot/pca/pca_plot_scratch.png" alt="pca-iris-scratch">
</p>
​
<p align="center"> 
    <img width="500" src="./temp/plot/pca/pca_plot_sklearn.png" alt="pca-iris-sklearn">
</p>

<p align="center"> 
    <img width="500" src="./temp/plot/pca/pca_variance_scratch.png" alt="pca-iris-elbow">
</p>
​