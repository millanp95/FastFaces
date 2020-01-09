# FastFaces
A Pytorch Implementation of a compressed Deep Learning Architecture using Low-Rank Factorization. The goal is to speed up and compress a network by eliminating redundancy in the 4D tensors that serve as convolutional kernels. The proposed tensor decomposition replaces the convolutional kernel with two consecutive kernels with lower rank as is shown in the figure. 


More specifically, if the convolutional layer have <a href="https://www.codecogs.com/eqnedit.php?latex=C" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C" title="C" /></a> inputs and <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a> outputs, 
