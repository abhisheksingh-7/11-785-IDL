import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # r_t
        self.affine_rx = self.Wrx @ self.x # h x 1
        self.linear_rx = self.affine_rx + self.brx
        self.affine_rh = self.Wrh @ self.hidden
        self.linear_rh = self.affine_rh + self.brh
        self.linear_rt = self.linear_rx + self.linear_rh
        self.r = self.r_act.forward(self.linear_rt)

        # z_t
        self.affine_zx = self.Wzx @ self.x
        self.linear_zx = self.affine_zx + self.bzx
        self.affine_zh = self.Wzh @ self.hidden
        self.linear_zh = self.affine_zh + self.bzh
        self.linear_zt = self.linear_zx + self.linear_zh
        self.z = self.z_act.forward(self.linear_zt)

        # n_t
        self.affine_nx = self.Wnx @ self.x
        self.linear_nx = self.affine_nx + self.bnx
        self.affine_nh = self.Wnh @ self.hidden
        self.linear_nh = self.affine_nh + self.bnh
        self.rt_nh = self.r * self.linear_nh
        self.linear_nt = self.linear_nx + self.rt_nh
        self.n = self.h_act.forward(self.linear_nt)

        # h_t
        self.z1 = (1 - self.z) * self.n
        self.z2 = self.z * self.hidden
        h_t = self.z1 + self.z2

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        print("delta shape")
        print(delta.shape)

        print("x shape")
        print(self.x.shape)

        print("hidden shape")
        print(self.hidden.shape)
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.
        # # ADDITIONAL TIP:
        # # Make sure the shapes of the calculated dWs and dbs  match the
        # # initalized shapes accordingly
        x_prime = self.x.reshape((-1, 1))
        print("x_prime shape")
        print(x_prime.shape)
        hidden_prime = self.hidden.reshape((-1, 1))
        print("hidden_prime shape")
        print(hidden_prime.shape)

        dLdh_t = delta.reshape((1, -1))
        print("dLdh_t shape")
        print(dLdh_t.shape)

        # h_t = self.z1 + self.z2
        dLdz1 = dLdz2 = dLdh_t # 1 x h
        print("dLdz1 shape")
        print(dLdz1.shape)
        print("dLdz2 shape")
        print(dLdz2.shape)

        # self.z2 = self.z * self.hidden
        dLdz = dLdz2 * self.hidden.reshape((1, -1)) # 1 x h
        print("dLdz shape")
        print(dLdz.shape)
        dh_prev_t = dLdz2 * self.z.reshape((1, -1)) # 1 x h
        print("dh_prev_t shape")
        print(dh_prev_t.shape)

        # self.z1 = (1 - self.z) * self.n
        dLdz += -1 * dLdz1 * self.n.reshape((1, -1)) # 1 x h
        print("dLdz shape")
        print(dLdz.shape)
        dLdn = dLdz1 * (1 - self.z).reshape((1, -1)) # 1 x h
        print("dLdn shape")
        print(dLdn.shape)

        # self.n = self.h_act.forward(self.linear_nt)
        dLdlinear_nt = dLdn * self.h_act.backward() # 1 x h
        print("dLdlinear_nt shape")
        print(dLdlinear_nt.shape)

        # self.linear_nt = self.linear_nx + self.rt_nh
        dLdlinear_nx = dLdrt_nh = dLdlinear_nt # 1 x h
        print("dLdlinear_nx shape")
        print(dLdlinear_nx.shape)
        print("dLdrt_nh shape")
        print(dLdrt_nh.shape)
        

        # self.rt_nh = self.r * self.linear_nh
        dLdr = dLdrt_nh * self.linear_nh.reshape((1, -1)) # 1 x h
        print("dLdr shape")
        print(dLdr.shape)
        dLdlinear_nh = dLdrt_nh * self.r.reshape((1, -1)) # 1 x h
        print("dLdlinear_nh shape")
        print(dLdlinear_nh.shape)

        # self.linear_nh = self.affine_nh + self.bnh
        dLdaffine_nh = self.dbnh = dLdlinear_nh # 1 x h
        print("dLdaffine_nh shape")
        print(dLdaffine_nh.shape)
        self.dbnh = self.dbnh.reshape((-1,))
        print("self.dbnh shape")
        print(self.dbnh.shape)

        # self.affine_nh = self.Wnh @ self.hidden
        self.dWnh += (hidden_prime @ dLdaffine_nh).T # h x h
        print("dWnh shape")
        print(self.dWnh.shape)
        dh_prev_t += dLdaffine_nh @ (self.Wnh) # 1 x h
        print("dh_prev_t shape")
        print(dh_prev_t.shape)

        # self.linear_nx = self.affine_nx + self.bnx
        dLdaffine_nx = self.dbnx = dLdlinear_nx # 1 x h
        print("dLdaffine_nx shape")
        print(dLdaffine_nx.shape)
        self.dbnx = self.dbnx.reshape((-1,))
        print("dbnx shape")
        print(self.dbnx.shape)

        # self.affine_nx = self.Wnx @ self.x
        self.dWnx += (x_prime @ dLdaffine_nx).T # h x i
        print("self.dWnx shape")
        print(self.dWnx.shape)
        dx = dLdaffine_nx @ self.Wnx # 1 x i
        print("dx shape")
        print(dx.shape)

        # self.z = self.z_act.forward(self.linear_zt)
        dLdlinear_zt = dLdz * self.z_act.backward() # 1 x h
        print("dLdlinear_zt shape")
        print(dLdlinear_zt.shape)

        # self.linear_zt = self.linear_zx + self.linear_zh
        dLdlinear_zx = dLdlinear_zh = dLdlinear_zt # 1 x h
        print("dLdlinear_zx shape")
        print(dLdlinear_zx.shape)
        print("dLdlinear_zh shape")
        print(dLdlinear_zh.shape)

        # self.linear_zh = self.affine_zh + self.bzh
        dLdaffine_zh = self.dbzh = dLdlinear_zh # 1 x h
        print("dLdlinear_zx shape")
        print(dLdlinear_zx.shape)
        self.dbzh = self.dbzh.reshape((-1,))
        print("self.dbzh shape")
        print(self.dbzh.shape)

        # self.affine_zh = self.Wzh @ self.hidden
        self.dWzh += (hidden_prime @ dLdaffine_zh).T # h x h
        print("self.dWzh shape")
        print(self.dWzh.shape)
        dh_prev_t += dLdaffine_zh @ self.Wzh # 1 x h
        print("dh_prev_t shape")
        print(dh_prev_t.shape)

        # self.linear_zx = self.affine_zx + self.bzx
        dLdaffine_zx = self.dbzx = dLdlinear_zx # 1 x h
        print("dLdaffine_zx shape")
        print(dLdaffine_zx.shape)
        self.dbzx = self.dbzx.reshape((-1,))
        print("self.dbzx shape")
        print(self.dbzx.shape)

        # self.affine_zx = self.Wzx @ self.x
        self.dWzx += (x_prime @ dLdaffine_zx).T # h x i
        print("self.dWzx shape")
        print(self.dWzx.shape)
        dx += dLdaffine_zx @ self.Wzx # 1 x i
        print("dx shape")
        print(dx.shape)

        # self.r = self.r_act.forward(self.linear_rt)
        dLdlinear_rt = dLdr * self.r_act.backward() # 1 x h
        print("dLdlinear_rt shape")
        print(dLdlinear_rt.shape)

        # self.linear_rt = self.linear_rx + self.linear_rh
        dLdlinear_rx = dLdlinear_rh = dLdlinear_rt
        print("dLdlinear_rx shape")
        print(dLdlinear_rx.shape)
        print("dLdlinear_rh shape")
        print(dLdlinear_rh.shape)

        # self.linear_rh = self.affine_rh + self.brh
        dLdaffine_rh = self.dbrh = dLdlinear_rh # 1 x h
        print("dLdaffine_rh shape")
        print(dLdaffine_rh.shape)
        self.dbrh = self.dbrh.reshape((-1,))
        print("self.dbrh shape")
        print(self.dbrh.shape)

        # self.affine_rh = self.Wrh @ self.hidden
        self.dWrh = (hidden_prime @ dLdaffine_rh).T # h x h
        print("self.dWrh shape")
        print(self.dWrh.shape)
        dh_prev_t += dLdaffine_rh @ self.Wrh # 1 x h
        print("dh_prev_t shape")
        print(dh_prev_t.shape)

        # self.linear_rx = self.affine_rx + self.brx
        dLdaffine_rx = self.dbrx = dLdlinear_rx # 1 x h
        print("dLdaffine_rx shape")
        print(dLdaffine_rx.shape)
        self.dbrx = self.dbrx.reshape((-1,))
        print("self.dbrx shape")
        print(self.dbrx.shape)

        # self.affine_rx = self.Wrx @ self.x
        self.dWrx = (x_prime @ dLdaffine_rx).T # h x i
        print("self.dWrx shape")
        print(self.dWrx.shape)
        dx += dLdaffine_rx @ self.Wrx # 1 x i
        print("dx shape")
        print(dx.shape)
        
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
