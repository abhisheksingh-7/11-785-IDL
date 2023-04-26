import numpy as np
from activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

        # misc
        self.Ones = None

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(Wihxt + bih + Whhht−1 + bhh) 
        """
        h_t = self.activation.forward((x @ np.transpose(self.W_ih)) + (np.ones((x.shape[0], 1)) @ self.b_ih.reshape(1, -1)) + (h_prev_t @ np.transpose(self.W_hh)) + (np.ones((x.shape[0], 1)) @ self.b_hh.reshape(1, -1)))
        assert h_t.shape == (x.shape[0], self.hidden_size)
        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h_t: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size, hidden_size = delta.shape
        batch_size, input_size = h_prev_l.shape

        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        dz = delta * self.activation.backward(state=h_t)
        assert dz.shape == delta.shape

        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += np.divide((np.transpose(dz) @ h_prev_l), batch_size)
        assert self.dW_ih.shape == (hidden_size, input_size)
        self.dW_hh += np.divide((np.transpose(dz) @ h_prev_t), batch_size)
        assert self.dW_hh.shape == (hidden_size, hidden_size)
        self.db_ih += np.divide((np.transpose(dz) @ np.ones((batch_size,1))).reshape(-1), batch_size)
        assert self.db_ih.shape == (hidden_size, )
        self.db_hh += np.divide((np.transpose(dz) @ np.ones((batch_size,1))).reshape(-1), batch_size)
        assert self.db_ih.shape == (hidden_size, )

        # # 2) Compute dx, dh_prev_t
        dx = dz @ self.W_ih
        assert dx.shape == (batch_size, input_size)
        dh_prev_t = dz @ self.W_hh
        assert dh_prev_t.shape == (batch_size, hidden_size)

        # 3) Return dx, dh_prev_t
        return dx, dh_prev_t