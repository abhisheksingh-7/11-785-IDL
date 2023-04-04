# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size) = N x C_in x W_in
        Return:
            Z (np.array): (batch_size, out_channels, output_size) = N x C_out X W_out
        """
        self.A = A
        batch_size = A.shape[0]
        input_size = A.shape[2]
        out_channels = self.W.shape[0]
        in_channels = self.W.shape[1]

        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size))

        # each Z[:, :, i] is tensordot of A[:, :, i:i+K] and W[:, :, :]
        # => each Z instance = np.tensordot((N, C_in, K), (C_out, C_in, K)),
        # to collapse C_in and K, axes = ([1,2], [1,2]) => Z[, , i] = N x C_out
        for i in range(output_size):
            # compute affine value
            Z[:, :, i] = np.tensordot(self.A[:, :, i:i+self.kernel_size], self.W, axes=((1,2), (1,2))) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        output_size = dLdZ.shape[2]
        input_size = self.A.shape[2]

        # dLdb
        # we get dLdb by just summing the elements of dLdZ channel wise
        # dLdb is a vector of shape equal to the number of output channels
        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        # dLdW
        # do a row-wise convolution between dLdZ and the input A to get dLdW
        for i in range(self.kernel_size):
            self.dLdW[:, :, i] = np.tensordot(dLdZ, self.A[:, :, i:i+output_size], axes=((0,2), (0,2))) 

        # dLdA
        dLdA = np.zeros(self.A.shape)
        # broadcast dLdZ C_in times : done implicitly
        # pad with K-1 zeros on both sides in the W_out dimension
        # as we require o/p to be larget than i/p as conv reduces size
        dLdZ = np.pad(dLdZ, ((0,0), (0,0), (self.kernel_size-1,self.kernel_size-1)), mode='constant')
        
        # flip each channel of filter left to right
        # W = C_out x C_in x K, filter on axis K
        W = np.flip(self.W, axis=2)

        # Convolve each flipped channel of the filter with 
        # the broadcasted and padded dLdZ to get dLdA
        # dLdZ-padded[i:i+K] = N x C_out x K
        # W = C_out x C_in x K
        # dLdA = N x C_in x W_in
        # dLdA = dLdZ-padded ⨂ W, axes=((1,2), (0,2))
        for i in range(input_size):
            dLdA[:, :, i] = np.tensordot(dLdZ[:, :, i:i+self.kernel_size], W, axes=((1,2), (0,2)))
        
        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)
        self.downsample1d = Downsample1d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z_conv = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z_conv)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ_ds = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ_ds)

        return dLdA
