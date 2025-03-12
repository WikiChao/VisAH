import math
import copy
from inspect import isfunction
from typing import Optional, List
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn, Tensor


class PositionalEncoding(nn.Module):
    """Positional encoding. See section 3.5 in https://arxiv.org/pdf/1706.03762"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

    def forward_temporal(self, x):
        x = x + self.pe[:, : x.size(2), :].transpose(-1, -2).unsqueeze(-2)
        return self.dropout(x)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def exists(val):
    return val is not None


def uniq(arr):
    """
    Return unique elements of an array
    """
    return {el: True for el in arr}.keys()


def default(val, d):
    """
    Return default value if value is None
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    """
    Return the maximum negative value of a tensor
    """
    return -torch.finfo(t.dtype).max


def init_(tensor):
    """
    Initialize a tensor with a normal distribution
    """
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    """
    Gated linear unit with GELU activation
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """
    Feedforward layer in a transformer
    """

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    """
    Normalize a tensor using group normalization
    """
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class CrossAttention(nn.Module):
    """
    Cross attention layer
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        :param x: input tensor of shape `[batch_size, sequence_length, channels]`
        :param context: context tensor of shape `[batch_size, context_length, channels]`
        :param mask (optional): attention mask of shape `[batch_size, heads, sequence_length, context_length]`
        :return: output tensor of shape `[batch_size, sequence_length, channels]`
        """
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~(mask == 1), max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """
    Basic transformer block, including self-attention, cross-attention and feedforward layer
    """

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=dim//n_heads,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        if context is None:
            return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)
        else:
            return checkpoint(
                self._forward, (x, context, mask), self.parameters(), self.checkpoint
            )

    def _forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class AV_Transformer(nn.Module):
    """
    Transformer block for sequence-like data.
    First, project the input (aka embedding)
    and add temporal position encoding.
    Then apply standard transformer action.
    Finally, project to output.
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        num_layers=3,
        context_dim=None,
    ):
        super().__init__()

        self.context_dim = context_dim

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv1d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=inner_dim,
                    checkpoint=False,
                )
                for d in range(depth)
            ]
        )

        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, inner_dim)
            encoder_layer = TransformerEncoderLayer(d_model=inner_dim, nhead=n_heads, dim_feedforward=inner_dim*2, dropout=dropout, activation="gelu", normalize_before=True)
            self.context_transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None)

        self.proj_out = zero_module(
            nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

        # self.proj_out = zero_module(
        #     nn.Linear(inner_dim, in_channels)
        # )

        self.pe = PositionalEncoding(in_channels)
        self.pe_context = PositionalEncoding(inner_dim)

    def forward(self, x, context=None, mask=None, input_position_encoding=False):
        """
        :param x: input tensor of shape `[batch_size, channels, sequence_length]`
        :param context: context tensor of shape `[batch_size, context_length, channels]`
        :param mask (optional): attention mask of shape `[batch_size, heads, sequence_length, context_length]`
        :param input_position_encoding: whether to apply positional encoding to the input tensor
        :return: output tensor of shape `[batch_size, channels, sequence_length]`
        """
        # note: if no context is given, cross-attention defaults to self-attention

        b, c, t = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c t -> b t c")
        if input_position_encoding:
            x = self.pe(x)
        if self.context_dim is not None:
            context = self.context_proj(context)
            context = self.context_transformer_encoder(context, mask=mask, src_key_padding_mask=None, pos=self.pe_context.pe[:, :context.shape[1], :])
        else:
            context = None
        # context = self.pe_context(context)
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)
        x = rearrange(x, "b t c -> b c t")
        x = self.proj_out(x)
        return x + x_in
