import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        
        # compute Z's dimensions
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        # initialize Z
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                # compute affine values and add bias
                Z[:, :, i, j] = np.tensordot(self.A[:, :, i:i+self.kernel_size, j:j+self.kernel_size], self.W, axes=((1,2,3),(1,2,3))) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        output_height = dLdZ.shape[2]
        output_width = dLdZ.shape[3]
        input_height = self.A.shape[2]
        input_width = self.A.shape[3]

        #dLdb
        self.dLdb = np.sum(dLdZ, axis=(0,2,3))

        #dLdW
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.dLdW[:, :, i, j] = np.tensordot(dLdZ, self.A[:, :, i:i+output_height, j:j+output_width], axes=((0,2,3),(0,2,3)))
        
        #dLdA
        dLdA = np.zeros(self.A.shape)
        dLdZ = np.pad(dLdZ, ((0,0), (0,0), (self.kernel_size-1,self.kernel_size-1), (self.kernel_size-1,self.kernel_size-1)), mode='constant')
        W = np.flip(self.W, axis=(2,3))
        for i in range(input_height):
            for j in range(input_width):
                dLdA[:, :, i, j] = np.tensordot(dLdZ[:, :, i:i+self.kernel_size, j:j+self.kernel_size], W, axes=((1,2,3), (0,2,3)))


        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels, out_channels=out_channels,
                 kernel_size=kernel_size, weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        Z_conv = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z_conv)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ_ds = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ_ds)

        return dLdA
