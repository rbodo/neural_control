from typing import List

import torch
from torch import nn, Tensor


class AntisymmetricRNN(torch.jit.ScriptModule):
    def __init__(self, input_dim, n_units=32, eps=0.01, gamma=0.01,
                 use_gating=True, init_W_std=1):
        super(AntisymmetricRNN, self).__init__()

        if use_gating:
            self.cell = AntisymmetricGatingRNNCell(input_dim, n_units, eps,
                                                   gamma, init_W_std)
        else:
            self.cell = AntisymmetricRNNCell(input_dim, n_units, eps, gamma,
                                             init_W_std)

        self.n_units = n_units

    @torch.jit.script_method
    def forward(self, x, h):
        # T = x.shape[1]
        x_ = x.unbind(1)
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(len(x_)):
            # h = self.cell(x[:, t, :], h)
            h = self.cell(x_[t], h)
            outputs += [h]
        return torch.stack(outputs), h


class AntisymmetricRNNCell(torch.jit.ScriptModule):
    def __init__(self, input_dim, n_units, eps, gamma, init_W_std=1):
        super(AntisymmetricRNNCell, self).__init__()

        # init Vh
        normal_sampler_V = torch.distributions.Normal(0, 1 / input_dim)
        self.Vh_weight = nn.Parameter(
            normal_sampler_V.sample((input_dim, n_units)))
        self.Vh_bias = nn.Parameter(torch.zeros(n_units))

        # init W
        normal_sampler_W = torch.distributions.Normal(0, init_W_std / n_units)
        self.W = nn.Parameter(normal_sampler_W.sample((n_units, n_units)))

        # init diffusion
        self.gamma_I = nn.Parameter(torch.eye(n_units, n_units) * gamma,
                                    requires_grad=False)

        self.eps = nn.Parameter(torch.Tensor([eps]), requires_grad=False)

    @torch.jit.script_method
    def update(self, x, h):
        # (W - WT - gammaI)h
        WmWT_h = torch.matmul(h, (self.W - self.W.transpose(1, 0) -
                                  self.gamma_I))
        # Vhx + bh
        Vh_x = torch.matmul(x, self.Vh_weight) + self.Vh_bias

        # (W - WT - gammaI)h + Vhx + bh
        linear_transform = WmWT_h + Vh_x

        # tanh((W - WT - gammaI)h + Vhx + bh)
        f = torch.tanh(linear_transform)

        # RHS of eq. 12
        return h + self.eps * f

    @torch.jit.script_method
    def forward(self, x, h):
        h = self.update(x, h)
        return h


class AntisymmetricGatingRNNCell(AntisymmetricRNNCell):
    def __init__(self, input_dim, n_units, eps, gamma, init_W_std):
        super(AntisymmetricGatingRNNCell, self).__init__(
            input_dim, n_units, eps, gamma, init_W_std)

        # init Vz
        normal_sampler_V = torch.distributions.Normal(0, 1 / input_dim)
        self.Vz_weight = nn.Parameter(
            normal_sampler_V.sample((input_dim, n_units)))
        # init input gate open by setting bias term equal to 1
        self.Vz_bias = nn.Parameter(torch.ones(n_units))

    @torch.jit.script_method
    def forward(self, x, h):
        WmWT_h = torch.matmul(h, (self.W - self.W.transpose(1, 0) -
                                  self.gamma_I))

        # Vhx + bh
        Vh_x = torch.matmul(x, self.Vh_weight) + self.Vh_bias

        # (W - WT - gammaI)h + Vhx + bh
        linear_transform1 = WmWT_h + Vh_x

        # Vzx + bz
        Vz_x = torch.matmul(x, self.Vz_weight) + self.Vz_bias

        # (W - WT - gammaI)h + Vzx + bz
        linear_transform2 = WmWT_h + Vz_x

        # tanh((W - WT - gammaI)h + Vh x + bh) *
        # sigm((W - WT - gammaI)h + Vz x + bz)
        f = torch.tanh(linear_transform1) * torch.sigmoid(linear_transform2)

        # eq. 13
        h = h + self.eps * f

        return h


class AntisymmetricRNNModel(torch.jit.ScriptModule):
    def __init__(self, input_dim, output_classes, n_units=32, eps=0.01,
                 gamma=0.01, use_gating=True, init_W_std=1, batch_size=128):
        super(AntisymmetricRNNModel, self).__init__()
        self.asrnn = AntisymmetricRNN(input_dim, n_units, eps, gamma,
                                      use_gating, init_W_std)
        self.fully_connected = torch.jit.trace(
            nn.Linear(n_units, output_classes),
            torch.randn(batch_size, n_units))

    @torch.jit.script_method
    def forward(self, x, h):
        output, h = self.asrnn(x, h)
        out = self.fully_connected(h)

        return out
