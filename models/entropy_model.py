import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

import torchac


class RoundNoGradient(torch.autograd.Function):
    """ TODO: check. """
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class UniverseQuantR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = np.random.uniform(-1,1)
        uniform_distribution = torch.distributions.Uniform(-0.5*torch.ones(x.size())*(2**b), 
                                                            0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):
        return g


class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x<1e-9] = 0
        except RuntimeError:
            print("ERROR! grad1[x<1e-9] = 0")
            grad1 = g.clone()
        pass_through_if = np.logical_or(x.cpu().detach().numpy() >= 1e-9, g.cpu().detach().numpy()<0.0)
        t = torch.Tensor(pass_through_if+0.0).to(grad1.device)

        return grad1*t


class EntropyBottleneck(nn.Module):
    """The layer implements a flexible probability density model to estimate
    entropy of its input tensor, which is described in this paper:
    >"Variational image compression with a scale hyperprior"
    > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
    > https://arxiv.org/abs/1802.01436"""
    
    def __init__(self, channels, init_scale=8, filters=(3,3,3)):
        """create parameters.
        """
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._channels = channels
        self.ASSERT = False
        # build.
        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        # Create variables.
        self._matrices = nn.ParameterList([])
        self._biases = nn.ParameterList([])
        self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):
            #
            self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix)
            self._matrices.append(self.matrix)
            #
            self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
            self.bias.data.copy_(init_bias)# copy or fill?
            self._biases.append(self.bias)
            #       
            self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            self.factor.data.fill_(0.0)
            self._factors.append(self.factor)

    def _logits_cumulative(self, inputs):
        """Evaluate logits of the cumulative densities.
        
        Arguments:
        inputs: The values at which to evaluate the cumulative densities,
            expected to have shape `(channels, 1, batch)`.

        Returns:
        A tensor of the same shape as inputs, containing the logits of the
        cumulatice densities evaluated at the the given inputs.
        """
        logits = inputs
        for i in range(len(self._filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i])
            logits = torch.matmul(matrix, logits)
            logits += self._biases[i]
            factor = torch.tanh(self._factors[i])
            logits += factor * torch.tanh(logits)
        
        return logits

    def _quantize(self, inputs, mode):
        """Add noise or quantize."""
        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "UniverseQuant":
            return UniverseQuantR.apply(inputs)
        if mode == "symbols":
            return RoundNoGradient.apply(inputs)

    def _likelihood(self, inputs, quantize_step=1.0):
        """Estimate the likelihood.
        inputs shape: [points, channels]
        """
        # reshape to (channels, 1, points)
        inputs = inputs.permute(1, 0).contiguous()# [channels, points]
        shape = inputs.size()# [channels, points]
        inputs = inputs.view(shape[0], 1, -1)# [channels, 1, points]
        inputs = inputs.to(self.matrix.device)
        if not isinstance(quantize_step, float):
            quantize_step = quantize_step.reshape(inputs.shape[0], 1, 1).to(inputs.device)# TODO
        # Evaluate densities.
        lower = self._logits_cumulative(inputs - 0.5 * quantize_step)
        upper = self._logits_cumulative(inputs + 0.5 * quantize_step)
        sign = -torch.sign(torch.add(lower, upper)).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        # reshape to (points, channels)
        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0)

        return likelihood

    def forward(self, inputs, quantize_mode="noise", scaler=None):
        """Pass a tensor through the bottleneck.
        
        Arguments:
        inputs: The tensor to be passed through the bottleneck.
        
        Returns:
            values: `Tensor` with the shape as `inputs` containing the perturbed
            or quantized input values.
            likelihood: `Tensor` with the same shape as `inputs` containing the
            likelihood of `values` under the modeled probability distributions.
        """
        if quantize_mode is None: outputs = inputs
        else: outputs = self._quantize(inputs, mode=quantize_mode)
        if scaler is None: quantize_step = 1.0
        else: quantize_step = 1/scaler.factor
        likelihood = self._likelihood(outputs, quantize_step=quantize_step)
        likelihood = Low_bound.apply(likelihood)

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    def _estimate_bitrate_from_pmf(self, pmf, sym):
        L = pmf.shape[-1]
        pmf = pmf.reshape(-1, L)
        sym = sym.reshape(-1, 1).to(pmf.device)
        assert pmf.shape[0] == sym.shape[0]
        relevant_probabilities = torch.gather(pmf, dim=1, index=sym.long())
        relevant_probabilities = relevant_probabilities.reshape(sym.shape)
        #bitrate = torch.sum(-torch.log2(relevant_probabilities.clamp(min=1e-3)))
        bitrate = torch.sum(-torch.log2(relevant_probabilities))
        return bitrate

    def get_quantized_symbols(self, min_v_list, max_v_list, quantize_step_list):
        """get quantized symbols according to the quantize step for each channel.
        """
        channels = len(min_v_list)
        symbols_list = []
        for min_v, max_v, quantize_step in zip(min_v_list, max_v_list, quantize_step_list):
            symbols = torch.arange(min_v, max_v+quantize_step, quantize_step)
            symbols_list.append(symbols)
        num_symbols = max([len(symbols) for _, symbols in enumerate(symbols_list)])
        symbols_array = torch.zeros(num_symbols, channels)
        for idx_ch, symbols in enumerate(symbols_list):
            symbols_array[:len(symbols), idx_ch] = symbols

        return symbols_array.to(min_v_list.device)

    @torch.no_grad()
    def compress(self, inputs, scaler=None):
        if scaler is None: 
            # quantize
            values = self._quantize(inputs, mode="symbols")
            # get symbols
            min_v = values.min().detach().float()
            max_v = values.max().detach().float()
            quantize_step = 1.0
            symbols = torch.arange(min_v, max_v+1)
            symbols = symbols.reshape(-1,1).repeat(1, values.shape[-1])# (num_symbols, channels)
            # get normalized values
            values_norm = values - min_v
            min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])
        else:
            # quantize
            values = inputs
            scaler = scaler.to(values.device)
            # get symbols
            min_v = values.min(dim=0)[0].detach().float()
            max_v = values.max(dim=0)[0].detach().float()
            quantize_step = 1.0/scaler.factor
            quantize_step = quantize_step.squeeze().detach().float()
            symbols = self.get_quantized_symbols(min_v, max_v, quantize_step)
            # get normalized values
            values_norm = scaler.encode(values).round()
            min_v_norm = scaler.encode(min_v).round()
            if True: assert (values_norm.min(dim=0)[0].round() == min_v_norm).all()
            values_norm = values_norm - min_v_norm

        values_norm = values_norm.to(torch.int16)

        # get pmf
        pmf = self._likelihood(symbols, quantize_step=quantize_step)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        pmf = pmf.permute(1,0)# (channels, num_symbols)

        # check: estimate_bitrate_from_pmf
        if self.ASSERT:
            out_pmf = pmf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1)# (points, channels, num_symbols)
            estimated_bits = self._estimate_bitrate_from_pmf(out_pmf, values_norm)
            print('estimated_bits:', estimated_bits.detach().cpu().numpy())

        # get cdf
        cdf = self._pmf_to_cdf(pmf)
        # arithmetic encoding
        out_cdf = cdf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1).detach().cpu()
        strings = torchac.encode_float_cdf(out_cdf, values_norm.cpu(), check_input_bounds=True)
        if self.ASSERT:
            if not scaler is None: 
                values_norm_dec = torchac.decode_float_cdf(out_cdf, strings)
                assert values_norm_dec.equal(values_norm)
                values_norm_dec += min_v_norm
                values_dec = scaler.decode(values_norm_dec)
                assert values_dec.equal(values)

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy()


    @torch.no_grad()
    def decompress(self, strings, min_v, max_v, shape, channels, scaler=None):
        # to tensor.
        min_v = torch.tensor(min_v, dtype=torch.float32)
        max_v = torch.tensor(max_v, dtype=torch.float32)
        if scaler == None:
            # get symbols
            min_v, max_v = min_v[0], max_v[0]
            symbols = torch.arange(min_v, max_v+1)
            symbols = symbols.reshape(-1,1).repeat(1, channels)
            quantize_step = 1.0
        else:
            quantize_step = 1.0/scaler.factor
            quantize_step = quantize_step.squeeze().detach().cpu().float()
            symbols = self.get_quantized_symbols(min_v, max_v, quantize_step)

        # get pmf
        pmf = self._likelihood(symbols, quantize_step=quantize_step)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        pmf = pmf.permute(1,0)
        # get cdf
        cdf = self._pmf_to_cdf(pmf)
        # arithmetic decoding
        out_cdf = cdf.unsqueeze(0).repeat(shape[0], 1, 1).detach().cpu()
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        if scaler == None:
            values += min_v
        else:
            min_v_norm = scaler.encode(min_v).round()
            values += min_v_norm
            values = scaler.decode(values)

        return values