import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention, MemoryEfficientAttentionFlashAttentionOp
from typing import Callable, Union, Optional

class FlashSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.device = device
        self.dtype = dtype

        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # FlashAttention's causal flag (optional)
        self.causal = False

        # Move to appropriate device and dtype if specified
        if device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        # Determine the batch size, sequence length, and embedding dimension
        is_batched = x.dim() == 3  # (B, S, D) or (S, B, D)

        if not self.batch_first and is_batched:
            x = x.transpose(0, 1)  # (B, S, D) -> (S, B, D)

        if self.batch_first and is_batched:
            # Shape: (B, S, D)
            B, S, D = x.shape
        else:
            # Shape: (S, B, D)
            S, B, D = x.shape
            x = x.transpose(0, 1)  # (S, B, D) -> (B, S, D)

        # Perform QKV projection and reshape
        qkv = self.qkv_proj(x)  # (B, S, 3 * D)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, H, S, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, S, Hd)

        # FlashAttention expects shape: (B, H, S, Hd)
        attn_output = memory_efficient_attention(
            q, k, v,
            attn_bias=None,
            op=MemoryEfficientAttentionFlashAttentionOp,
        )  # Output: (B, H, S, Hd)

        # Reshape the attention output and apply the final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)  # (B, S, D)
        
        if not self.batch_first and is_batched:
            output = output.transpose(0, 1)  # back to (S, B, D)
            
            
        return self.out_proj(attn_output)




class FlashTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Args:
            d_model: The dimension of the input and output feature vectors.
            nhead: The number of attention heads.
            dim_feedforward: The dimension of the feedforward network.
            dropout: The dropout probability.
            activation: The activation function to use (default is ReLU).
            batch_first: If True, the input and output tensors have the shape (B, S, D).
            norm_first: If True, LayerNorm is applied before the attention and feedforward layers.
            bias: If True, bias terms are added to the linear layers.
            device: The device on which to allocate the tensors.
            dtype: The dtype of the tensors.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.activation = activation

        # FlashSelfAttention Layer
        self.self_attn = FlashSelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, **factory_kwargs)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor of shape (B, S, D) or (S, B, D) depending on batch_first
            src_mask: Optional mask for the attention mechanism.
        """
        # GPT 的抽象行为
        # if self.batch_first:
        #     # src shape: (B, S, D)
        #     residual = src
        #     if self.norm_first:
        #         src = self.norm1(src)
        # else:
        #     # src shape: (S, B, D)
        #     residual = src
        #     if self.norm_first:
        #         src = self.norm1(src)
        residual = src
        if self.norm_first:
            src = self.norm1(src)
        # Self-attention
        attn_output = self.self_attn(src)
        src = residual + self.dropout_layer(attn_output)

        if not self.norm_first:
            src = self.norm1(src)

        # Feedforward network
        residual = src
        src = self.norm2(src)
        src = self.linear2(self.dropout_layer(self.activation(self.linear1(src))))
        src = residual + self.dropout_layer(src)
        return src
        
        
class FlashTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, d_model] or [seq_len, batch_size, d_model]
            src_mask: Optional mask for attention [batch_size, seq_len] or [seq_len, seq_len]
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output