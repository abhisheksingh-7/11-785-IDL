import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        out_channels = in_channels # Z = N x C_in x H_out x W_out

        # initialize Z
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        self.max_index = np.empty(shape=Z.shape, dtype=tuple)

        for b in range(batch_size):
            for c in range(out_channels):
                for i, j in np.ndindex(output_width, output_height):
                    pool = A[b, c, i:i+self.kernel, j:j+self.kernel]
                    Z[b, c, i, j] = pool.max()
                    # https://stackoverflow.com/questions/3584243/get-the-position-of-the-largest-value-in-a-multi-dimensional-numpy-array
                    max_row, max_col = np.unravel_index(pool.argmax(), pool.shape)
                    self.max_index[b, c, i, j] = i + max_row, j + max_col

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1
        in_channels = out_channels

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for b in range(batch_size):
            for c in range(in_channels):
                for i, j in np.ndindex(output_width, output_height):
                    max_row, max_col = self.max_index[b, c, i, j]
                    dLdA[b, c, max_row, max_col] += dLdZ[b, c, i, j]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        out_channels = in_channels # Z = N x C_in x H_out x W_out

        # initialize Z
        Z = np.zeros((batch_size, out_channels, output_width, output_height))

        for b in range(batch_size):
            for c in range(out_channels):
                for i, j in np.ndindex(output_width, output_height):
                    pool = A[b, c, i:i+self.kernel, j:j+self.kernel]
                    Z[b, c, i, j] = pool.mean()
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1
        in_channels = out_channels

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for b in range(batch_size):
            for c in range(in_channels):
                for i, j in np.ndindex(output_width, output_height):
                    dLdA[b, c, i:i+self.kernel, j:j+self.kernel] += dLdZ[b, c, i, j] / self.kernel ** 2

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel=kernel)
        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # max pool
        Z = self.maxpool2d_stride1.forward(A)
        # downsample
        Z = self.downsample2d.forward(Z)
        return Z
        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # downsample
        dLdZ = self.downsample2d.backward(dLdZ)
        # max pool
        dLdA = self.maxpool2d_stride1.backward(dLdZ)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel=kernel)
        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # mean pool
        Z = self.meanpool2d_stride1.forward(A)
        # downsample
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # downsample
        dLdZ = self.downsample2d.backward(dLdZ)
        # mean pool
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA

