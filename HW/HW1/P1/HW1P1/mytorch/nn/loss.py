import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (A - Y) * (A - Y)
        sse = np.ones((1, self.N)) @ se @ np.ones((self.C, 1))
        assert sse.shape == (1, 1)
        mse = sse / (2*self.N*self.C)

        return mse[0][0]

    def backward(self):

        dLdA = (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C, 1))
        Ones_N = np.ones((self.N, 1))

        exp_A = np.exp(self.A)
        row_sum = np.reshape(np.sum(exp_A, axis=1), (self.N,1))
        
        self.softmax = exp_A / row_sum
        assert self.softmax.shape == self.A.shape
        
        crossentropy = (-self.Y * np.log(self.softmax)) @ Ones_C
        assert crossentropy.shape == (self.N, 1)

        sum_crossentropy = np.transpose(Ones_N) @ crossentropy 
        L = sum_crossentropy / self.N

        return L[0][0]

    def backward(self):

        dLdA = self.softmax - self.Y

        return dLdA
