
<h1 align="center">ML Workbook</h1>

## Description
These are my initial footsteps into Machine Learning.
I have tried to implement ML/DL algorithms from scratch using numpy, because getting your hands dirty is the best way to learn.


## Comparision of optimization Algorithms on saddle point
[Source code](./src/misc/optimizer_comparision.py)

<table>
    <tr>
        <td><p align="center">Scratch optimizers</p></td>
        <td><p align="center">Tensorflow optimizers</p></td>
    </tr>
    <tr>
        <td><img width="80%" src="./temp/plot/nn/optimizers/optimizers_comparision_scratch.gif" alt="Scratch optimizers"></td>
        <td><img width="80%" src="./temp/plot/nn/optimizers/optimizers_comparision_tf.gif" alt="Tensorflow optimizers"></td>
    </tr>
</table>


## Comparision of optimization Algorithms on MNIST dataset
[Source code](./src/dl/example.py)

I have compared the performance of a Convolutional Neural Network with different optimizers.
Models used same test and train datasets. Here are the results.
​

<table>
    <tr>
        <td><p align="center">Gradient Descent Accuracy</p></td>
        <td><p align="center">Gradient Descent Loss</p></td>
    </tr>
    <tr>
        <td><img width="500" src="./temp/plot/nn/mnist/gd/accuracy.png" alt="Gradient Descent"></td>
        <td><img width="500" src="./temp/plot/nn/mnist/gd/loss.png" alt="Gradient Descent"></td>
    </tr>
    <tr>
        <td><p align="center">Momentum Accuracy</p></td>
        <td><p align="center">Momentum Loss</p></td>
    </tr>
    <tr>
        <td><img width="500" src="./temp/plot/nn/mnist/momentum/accuracy.png" alt="Momentum Optimizer"></td>
        <td><img width="500" src="./temp/plot/nn/mnist/momentum/loss.png" alt="Momentum Optimizer"></td>
    </tr>
    <tr>
        <td><p align="center">RMSProp Accuracy</p></td>
        <td><p align="center">RMSProp Loss</p></td>
    </tr>
    <tr>
        <td><img width="500" src="./temp/plot/nn/mnist/rmsprop/accuracy.png" alt="RMSProp Optimizer"></td>
        <td><img width="500" src="./temp/plot/nn/mnist/rmsprop/loss.png" alt="RMSProp Optimizer"></td>
    </tr>
    <tr>
        <td><p align="center">Adam Accuracy</p></td>
        <td><p align="center">Adam Loss</p></td>
    </tr>
    <tr>
        <td><img width="500" src="./temp/plot/nn/mnist/adam/accuracy.png" alt="Adam Optimizer"></td>
        <td><img width="500" src="./temp/plot/nn/mnist/adam/loss.png" alt="Adam Optimizer"></td>
    </tr>
</table>


## Comparision of some common ML classification Algorithms 
[Source code](./src/misc/classifier_comparision.py)

I have tried to compare the performance of some common ML Algorithms on sklearn datasets.
I have also compared Sklearn's implementation with my scratch implementation side by side.

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
