import numpy as np

class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))
        assert self.W.shape == (out_features, in_features)
        self.b = np.zeros(out_features)
        assert self.b.shape == (out_features,)

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]  # store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))

        # Z = (A . W^T) + (Ones . B^T)
        Z = (self.A @ np.transpose(self.W)) + (self.Ones @ np.transpose(self.b))
        assert Z.shape == (self.N, self.W.shape[0])

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: how changes in the layer's output affect loss
        :return dLdA: how changes in the layer's inputs affect loss
        """
        assert dLdZ.shape == (self.A.shape[0], self.W.shape[0])
        dZdA = np.transpose(self.W)
        assert dZdA.shape == np.transpose(self.W).shape
        dZdW = self.A
        assert dZdW.shape == self.A.shape
        dZdb = self.Ones
        assert dZdb.shape == (self.A.shape[0],1)

        dLdA = dLdZ @ np.transpose(dZdA)
        assert dLdA.shape == self.A.shape
        dLdW = np.transpose(dLdZ) @ self.A
        assert dLdW.shape == self.W.shape
        dLdb = np.transpose(dLdZ) @ dZdb
        assert dLdb.shape == (self.W.shape[0],1)

        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
