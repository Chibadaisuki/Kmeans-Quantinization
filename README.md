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
