import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size = A.shape[0]
        in_channels = A.shape[1]
        input_width = A.shape[2]
        output_width = self.upsampling_factor * (input_width - 1) + 1
        
        Z = np.zeros((batch_size, in_channels, output_width))
        
        Z[:, :, ::self.upsampling_factor] = A
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        self.input = A

        Z = A[:, :, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size = dLdZ.shape[0]
        in_channels = dLdZ.shape[1]
        output_width = dLdZ.shape[2]
        input_width = self.input.shape[2]

        # gradient of input is same size as input => self.input.shape[2]
        dLdA = np.zeros((batch_size, in_channels, input_width))

        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size = A.shape[0]
        in_channels = A.shape[1]
        input_height = A.shape[2]
        input_width = A.shape[3]

        output_height = (self.upsampling_factor * (input_height - 1) + 1)
        output_width = (self.upsampling_factor * (input_width - 1) + 1)

        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.input = A

        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size = dLdZ.shape[0]
        in_channels = dLdZ.shape[1]
        output_height = dLdZ.shape[2]
        output_width = dLdZ.shape[3]

        input_height = self.input.shape[2]
        input_width = self.input.shape[3]

        # gradient of input is same size as input => self.input.shape[2],  self.input.shape[3]
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))

        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA