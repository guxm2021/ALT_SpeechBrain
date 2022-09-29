import torch
import torch.nn as nn

from torch import Tensor


class QuasiRNNLayer(torch.nn.Module):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an
    input sequence.

    Arguments
    ---------
    input_size : int
        The number of expected features in the input x.
    hidden_size : int
        The number of features in the hidden state h. If not specified,
        the input size is used.
    zoneout : float
        Whether to apply zoneout (i.e. failing to update elements in the
        hidden state) to the hidden state updates. Default: 0.
    output_gate : bool
        If True, performs QRNN-fo (applying an output gate to the output).
        If False, performs QRNN-f. Default: True.

    Example
    -------
    >>> import torch
    >>> model = QuasiRNNLayer(60, 256, bidirectional=True)
    >>> a = torch.rand([10, 120, 60])
    >>> b = model(a)
    >>> b[0].shape
    torch.Size([10, 120, 512])
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional,
        zoneout=0.0,
        output_gate=True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.zoneout = zoneout
        self.output_gate = output_gate
        self.bidirectional = bidirectional

        stacked_hidden = (
            3 * self.hidden_size if self.output_gate else 2 * self.hidden_size
        )
        self.w = torch.nn.Linear(input_size, stacked_hidden, True)

        self.z_gate = nn.Tanh()
        self.f_gate = nn.Sigmoid()
        if self.output_gate:
            self.o_gate = nn.Sigmoid()

    def forgetMult(self, f, x, hidden):
        
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        result = []
        htm1 = hidden
        hh = f * x

        for i in range(hh.shape[0]):
            h_t = hh[i, :, :]
            ft = f[i, :, :]
            if htm1.shape[0] != 0: #is not None:
                h_t = h_t + (1 - ft) * htm1
            result.append(h_t)
            htm1 = h_t

        return torch.stack(result)

    def split_gate_inputs(self, y):
        
        if self.output_gate:
            z, f, o = y.chunk(3, dim=-1)
        else:
            z, f = y.chunk(2, dim=-1)
            o = None
        return z, f, o

    def forward(self, x, hidden=Tensor()): #None):
        
        """Returns the output of the QRNN layer.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.
        """
        if x.ndim == 4:
            # if input is a 4d tensor (batch, time, channel1, channel2)
            # reshape input to (batch, time, channel)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # give a tensor of shape (time, batch, channel)
        x = x.permute(1, 0, 2)
        if self.bidirectional:
            x_flipped = x.flip(0)
            x = torch.cat([x, x_flipped], dim=1)

        # note: this is equivalent to doing 1x1 convolution on the input
        y = self.w(x)

        z, f, o = self.split_gate_inputs(y)

        z = self.z_gate(z)
        f = self.f_gate(f)
        if o is not None:
            o = self.o_gate(o)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron
        # keeps the old value
        if self.zoneout:
            if self.training:
                mask = (
                    torch.empty(f.shape)
                    .bernoulli_(1 - self.zoneout)
                    .to(f.get_device())
                ).detach()
                f = f * mask
            else:
                f = f * (1 - self.zoneout)

        z = z.contiguous()
        f = f.contiguous()

        # Forget Mult
        c = self.forgetMult(f, z, hidden)

        # Apply output gate
        if o is not None:
            h = o * c
        else:
            h = c

        # recover shape (batch, time, channel)
        c = c.permute(1, 0, 2)
        h = h.permute(1, 0, 2)

        if self.bidirectional:
            h_fwd, h_bwd = h.chunk(2, dim=0)
            h_bwd = h_bwd.flip(1)
            h = torch.cat([h_fwd, h_bwd], dim=2)

            c_fwd, c_bwd = c.chunk(2, dim=0)
            c_bwd = c_bwd.flip(1)
            c = torch.cat([c_fwd, c_bwd], dim=2)

        return h, c[-1, :, :]


class QuasiRNN(nn.Module):
    """This is a implementation for the Quasi-RNN.

    https://arxiv.org/pdf/1611.01576.pdf

    Part of the code is adapted from:
    https://github.com/salesforce/pytorch-qrnn

    Arguments
    ---------
    hidden_size : int
        The number of features in the hidden state h. If not specified,
        the input size is used.
    input_shape : tuple
        The shape of an example input. Alternatively, use ``input_size``.
    input_size : int
        The size of the input. Alternatively, use ``input_shape``.
    num_layers : int
        The number of QRNN layers to produce.
    zoneout : bool
        Whether to apply zoneout (i.e. failing to update elements in the
        hidden state) to the hidden state updates. Default: 0.
    output_gate : bool
        If True, performs QRNN-fo (applying an output gate to the output).
        If False, performs QRNN-f. Default: True.

    Example
    -------
    >>> a = torch.rand([8, 120, 40])
    >>> model = QuasiRNN(
    ...     256, num_layers=4, input_shape=a.shape, bidirectional=True
    ... )
    >>> b, _ = model(a)
    >>> b.shape
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        hidden_size,
        input_shape=None,
        input_size=None,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        **kwargs,
    ):
        assert bias is True, "Removing underlying bias is not yet supported"
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout if dropout > 0 else None
        self.kwargs = kwargs

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size.")

        # Computing the feature dimensionality
        if input_size is None:
            if len(input_shape) > 3:
                self.reshape = True
            input_size = torch.prod(torch.tensor(input_shape[2:]))

        layers = []
        for layer in range(self.num_layers):
            layers.append(
                QuasiRNNLayer(
                    input_size
                    if layer == 0
                    else self.hidden_size * 2
                    if self.bidirectional
                    else self.hidden_size,
                    self.hidden_size,
                    self.bidirectional,
                    **self.kwargs,
                )
            )
        self.qrnn = torch.nn.ModuleList(layers)

        if self.dropout:
            self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, x, hidden=Tensor()):

        next_hidden = []

        for i, layer in enumerate(self.qrnn):
            # x, h = layer(x, None if hidden.shape[0] == 0 else hidden[i])
            if hidden.shape[0] == 0:
                x, h = layer(x)
            else:
                x, h = layer(x, hidden[i])

            next_hidden.append(h)
            
            if self.dropout is not None and i < len(self.qrnn) - 1:
                x = self.dropout(x)
        
        next_hidden = torch.stack(next_hidden, dim=0)
        
        return x#, next_hidden

# if __name__ == '__main__':
#     a = torch.rand([8, 120, 40])
#     model = QuasiRNN(256, num_layers=4, input_shape=a.shape, bidirectional=True)
#     b = model(a)
#     print(b.shape)