# FastFaces
A Pytorch Implementation of a compressed Deep Learning Architecture using Low-Rank Factorization. 

## About
The goal is to speed up and compress a network by eliminating redundancy in the 4D tensors that serve as convolutional kernels. The proposed tensor decomposition replaces the convolutional kernel with two consecutive kernels with lower rank as is shown in the figure. 

<div id="banner">
<div class="inline-block">
<img src ="Pictures\1.jpg">
</div>

<div class="inline-block">
<img src ="Pictures\2.jpg">
</div>
</div>

More specifically, if the convolutional layer have <a href="https://www.codecogs.com/eqnedit.php?latex=C" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C" title="C" /></a> inputs and <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a> outputs, the main objective is to find two intermediate kernels <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathcal{H}&space;\in&space;\mathbb{R}^{C&space;\times&space;d&space;\times&space;1&space;\times&space;K}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathcal{H}&space;\in&space;\mathbb{R}^{C&space;\times&space;d&space;\times&space;1&space;\times&space;K}$" title="$\mathcal{H} \in \mathbb{R}^{C \times d \times 1 \times K}$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{V}\in&space;\mathbb{R}^{K&space;\times&space;d&space;\times&space;1&space;\times&space;N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{V}\in&space;\mathbb{R}^{K&space;\times&space;d&space;\times&space;1&space;\times&space;N}" title="\mathcal{V}\in \mathbb{R}^{K \times d \times 1 \times N}" /></a> such that <a href="https://www.codecogs.com/eqnedit.php?latex=K&space;<&space;N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K&space;<&space;N" title="K < N" /></a>, constraining the rank of the layer to be <a href="https://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K" title="K" /></a>.


The new proposed tensor has the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{\mathcal{W}_{n}^{c}}=\sum_{k=1}^{K}&space;\mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{\mathcal{W}_{n}^{c}}=\sum_{k=1}^{K}&space;\mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T}" title="\tilde{\mathcal{W}_{n}^{c}}=\sum_{k=1}^{K} \mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T}" /></a>

and given that the convolution is distributive over addition,  we can produce the same number offeature maps performing two independent convolutions:

<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{\mathcal{W}}_{n}&space;*&space;\mathcal{Z}=\sum_{c=1}^{C}&space;\sum_{k=1}^{K}&space;\mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T}&space;*&space;\mathcal{Z}^{c}=\sum_{k=1}^{K}&space;\mathcal{H}_{n}^{k}&space;*\left(\sum_{c=1}^{C}&space;\mathcal{V}_{k}^{c}&space;*&space;\mathcal{Z}^{c}\right)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{\mathcal{W}}_{n}&space;*&space;\mathcal{Z}=\sum_{c=1}^{C}&space;\sum_{k=1}^{K}&space;\mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T}&space;*&space;\mathcal{Z}^{c}=\sum_{k=1}^{K}&space;\mathcal{H}_{n}^{k}&space;*\left(\sum_{c=1}^{C}&space;\mathcal{V}_{k}^{c}&space;*&space;\mathcal{Z}^{c}\right)." title="\tilde{\mathcal{W}}_{n} * \mathcal{Z}=\sum_{c=1}^{C} \sum_{k=1}^{K} \mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T} * \mathcal{Z}^{c}=\sum_{k=1}^{K} \mathcal{H}_{n}^{k} *\left(\sum_{c=1}^{C} \mathcal{V}_{k}^{c} * \mathcal{Z}^{c}\right)." /></a>

As the main goal is to maintain the overall accuracy, it is required to keep the filter products as close as possible to the original ones. The Frobenius norm can be used to define the objective function to be minimized, leading to the problem:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;&&space;\underset{(\mathcal&space;H,&space;\mathcal&space;V)}{\text{minimize}}&space;&&space;&&space;{E_{1}(\mathcal{H},&space;\mathcal{V})&space;:=\sum_{n,&space;c}\left\|\mathcal{W}_{n}^{c}-\sum_{k=1}^{K}&space;\mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T}\right\|_{F}^{2}}&space;\\&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&&space;\underset{(\mathcal&space;H,&space;\mathcal&space;V)}{\text{minimize}}&space;&&space;&&space;{E_{1}(\mathcal{H},&space;\mathcal{V})&space;:=\sum_{n,&space;c}\left\|\mathcal{W}_{n}^{c}-\sum_{k=1}^{K}&space;\mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T}\right\|_{F}^{2}}&space;\\&space;\end{aligned}" title="\begin{aligned} & \underset{(\mathcal H, \mathcal V)}{\text{minimize}} & & {E_{1}(\mathcal{H}, \mathcal{V}) :=\sum_{n, c}\left\|\mathcal{W}_{n}^{c}-\sum_{k=1}^{K} \mathcal{H}_{n}^{k}\left(\mathcal{V}_{k}^{c}\right)^{T}\right\|_{F}^{2}} \\ \end{aligned}" /></a>

Whic can be solved in a lower dimension space according to [1] using the following linear mapping

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;\mathcal{T}&space;:&space;\mathbb{R}^{C&space;\times&space;d&space;\times&space;d&space;\times&space;N}&space;&\rightarrow&space;\mathbb{R}^{C&space;d&space;\times&space;d&space;N}&space;\\&space;(i_1,&space;i_2,i_3,i_4)&space;&\mapsto&space;((i_1-1)d&plus;i_2,&space;\&space;\:&space;(j_4-1d&plus;i_3)),&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\mathcal{T}&space;:&space;\mathbb{R}^{C&space;\times&space;d&space;\times&space;d&space;\times&space;N}&space;&\rightarrow&space;\mathbb{R}^{C&space;d&space;\times&space;d&space;N}&space;\\&space;(i_1,&space;i_2,i_3,i_4)&space;&\mapsto&space;((i_1-1)d&plus;i_2,&space;\&space;\:&space;(j_4-1d&plus;i_3)),&space;\end{align*}" title="\begin{align*} \mathcal{T} : \mathbb{R}^{C \times d \times d \times N} &\rightarrow \mathbb{R}^{C d \times d N} \\ (i_1, i_2,i_3,i_4) &\mapsto ((i_1-1)d+i_2, \ \: (j_4-1d+i_3)), \end{align*}" /></a>

This mapping corresponds to the alignment of all the filters together in a matrix as is shown in the picture
 <p align="center">
  <img src="Pictures\2D Kernel.png">
</p>

Finally, the SVD decomposition of the matrix is computed and the product of the first <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> singular values and vectors will be mapped back to the original space. That is the solution to the original minimization problem. See [1] for a detailed proof.   

## Results. 
 <p align="center">
  <img src="Pictures\Picture.png">
</p>
The method was tested using the public the implementation of the Single Shot Multibox Detector (SSD) models provided by https://github.com/qfgaohao/pytorch-ssd. The method was tested  using a private dataset provided by the public transportation system in Bogotá Colombia with the aim of developing an efficient head count software. We were concerned about speedup and performance change for different layers. Performance change being relative to the baseline model with accuracy of 81.05 % and a total number of 6.601.011 parameters. The values in the weights reduction column represent the percentage of the original number of values in the particular layer that is used in the comprised model. 
 <p align="center">
  <img src="Pictures\Detection.png">
</p>

## Reference. 

[1].Cheng Tai, Tong Xiao, Xiaogang Wang, and E Weinan. Convolutional neural networks with low-rank regularization. CoRR, abs/1511.06067, 2016. 
