import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = np.sum(Z, axis=0) / self.N
        self.V = np.sum(np.square((self.Z - self.M)), axis=0) / self.N

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M) / (np.sqrt(self.V+self.eps))
            self.BZ = (self.BW * self.NZ) + self.Bb

            self.running_M = (self.alpha * self.running_M) + ((1 - self.alpha) * self.M)
            self.running_V = (self.alpha * self.running_V) + ((1 - self.alpha) * self.V)
        else:
            # inference mode
            self.NZ = (self.Z - self.running_M) / (np.sqrt(self.running_V+self.eps))
            self.BZ = (self.BW * self.NZ) + self.Bb

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum((dLdBZ * self.NZ), axis=0)
        self.dLdBb = np.sum(dLdBZ, axis=0)

        dLdNZ = dLdBZ * self.BW
        dLdV = -0.5 * np.sum(((dLdNZ * (self.Z - self.M)) * (np.power((self.V + self.eps), -1.5))), axis=0)
        dLdM = np.sum((dLdNZ * (-np.power((self.V + self.eps), -0.5)) - (0.5 * ((self.Z - self.M) * (np.power((self.V + self.eps), -1.5) * ((-2/self.N) * np.sum((self.Z - self.M), axis=0)))))), axis=0)
        dLdZ = (dLdNZ * np.power((self.V + self.eps), -0.5)) + (dLdV * ((2/self.N) * (self.Z - self.M))) + (dLdM / self.N)
        return dLdZ
