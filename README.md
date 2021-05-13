# Kmeans-Quantinization
KMeans with Loss Function

The new clustering algorithm is :
 argmini loss(x -> ci)  => argmini  (L(x)+ dL/dx (x - ci) )

To our understanding, x is each individual weight and L(x)  is the loss value that we got from the original network. Then, L(x) is the same for all weights. So the problem becomes a 1-d Kmeans clustering problem, with weighted L1 distance from weight to its closest centroids. 

Algorithm becomes: argmini   dL/dx* (x - ci)

Imaging in a 1-d axis, if dL/dx is positive, (x - ci) would tend to choose the largest farthest point as ci so that (x-ci) is negative with the largest absolute value. Similarly, if dL/dx is negative, (x - ci) would tend to choose the smallest farthest point as ci so that (x-ci) is positive with the largest absolute value. Then, the algorithm will also choose the two points. In order to avoid this, we need to use the absolute value to do all comparison. 

Algorithm becomes: argmini  | dL/dx *(x - ci)|

Hence, after clustering, we use similar algorithm to find cluster mean, 
argminc  x  Clusterloss (x -> c) =argminc x  Cluster| dL/dx *(x - c) |

The new centroids becomes the mean of x  Cluster| dL/dx *(x - c) | for each cluster. 


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Data](#generator)
	- [Model](#generator)
	- [Kmeans](#generator)
- [Contributors](#contributors)
- [License](#license)

## Background

With the exponential growth of computational power and the invention of various novel models, Deep Neural Network (DNN) has led to a series of breakthroughs in many different fields. Nowadays, DNNs could achieve groundbreaking performance in countless applications such as style transfer, classifications, haze removal and other applications. However, the deployment of these high-performance models onto resource-constrained devices (e.g., mobile devices with limited storage capacity and computation capability) hinders usage due to their high storage complexity. For instance, ResNet needs more than $10^7$ bits for weight storage to achieve an accuracy of 70\% on the ImageNet dataset.

Neural network quantization provides a promising solution to this problem since it could drastically reduce the model size while preserving most network performance. Quantization is an active area of research and development, enabling neural networks to be practically usable and more efficient in various domains. Out of all the quantization paradigms, bitwise quantization is one of the most successful and prominent approaches to quantization. Instead of designing novel network architectures that exploits memory-efficient operations, bitwise quantization targets parameters of existing models and reduce them from 32bit floating point into lower bit-depth representations. However, the performance of bitwise quantization methods depends on many attributes of the model, such as depth and width. Furthermore, these bitwise quantization methods use stable mapping functions and mapping intervals, leading to their lack of generality and potential waste of scarce bits. 

In this project, we proposed a novel quantization method based on k-means clustering. Given the network parameters, our approach will generate problem-specific k-mean clustering that minimizes the loss of the model based on the training gradient of the network. Our method will then quantize the entire model by remapping all the network parameters to their corresponding centroids. 

## Install

This project uses pytorch and jyupter notebook. Go check them out if you don't have them locally installed.


## Usage

This is only a documentation package.


### Data

To get the data for the model.

### Model

Toy model of MLP and CNN

### Kmeans

Kmeans method for quantization


### Contributors

This project exists thanks to all the people who contribute. 

## License

[MIT](LICENSE)
