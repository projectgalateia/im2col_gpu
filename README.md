im2col_gpu
==========

MATLAB im2col implementation on GPU mex

A mex source code written in CUDA.

Takes N images with dimension of mm x nn x c and column size m x n.
Then outputs im2col applied, (mm-m+1) x (nn-n+1) x N images with dimension of m x n x c.
