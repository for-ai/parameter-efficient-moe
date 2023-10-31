"""LoRA implementation"""

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning
from flaxformer.types import Array, Initializer, DType
from flaxformer.components import dense
from src import routing


class LoRA(nn.Module):
  """LoRA implementation
  
  Attributes:
    router: Router class
    rank: LoRA rank
    alpha = LoRA aplha
    lora_init_A: LoRA A initializer
    lora_init_B: LoRA B initializer
    lora_axis_names_A: Sharding axis names for LoRA A
    lora_axis_names_B: Sharding axis names for LoRA B
    dtype: Activation dtype
    output_dim: LoRA output dimensions
  """
  rank: int = 2  
  lora_init_A: Initializer = nn.initializers.normal(stddev=2e-2)
  lora_init_B: Initializer = nn.initializers.zeros
  lora_axis_names_A: Sequence[str] = ('mlp', 'rank')
  lora_axis_names_B: Sequence[str] = ('rank', 'mlp')
  alpha = 16
  output_dim: Optional[int] = None
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x: Array, **kwargs) -> Array:

    *rest, hidden = x.shape    

    x = jax.lax.convert_element_type(x, self.dtype)

    #[hidden, rank]
    lora_a = partitioning.param_with_axes(
              'lora_A',
              self.lora_init_A,
              (hidden, self.rank),
              jnp.float32,
              axes=self.lora_axis_names_A)
    
    lora_a = jax.lax.convert_element_type(lora_a, self.dtype)

    #[batch, seq_len, rank]
    ax = jnp.einsum('...d,dr->...r', 
                         x,
                         lora_a)                        

    # Add expert axis name to the partitioning axes
    ax = partitioning.with_sharding_constraint(ax, ('batch', 'length', 'unmodeled'))
    ax = jax.lax.convert_element_type(ax, self.dtype)

    #[rank, hidden]
    lora_b = partitioning.param_with_axes(
              'lora_B',
              self.lora_init_B,
              (self.rank, (self.output_dim if self.output_dim else hidden)),
              jnp.float32,
              axes=self.lora_axis_names_B)
    
    lora_b = jax.lax.convert_element_type(lora_b, self.dtype)
    
    #[batch, seq_len, rank]
    bax = jnp.einsum('...r,rd->...d', 
                         ax,
                         lora_b)  

    return bax * (self.alpha / self.rank)


class LoRAAttention(nn.Module):
  """LoRA implementation for Attention class
  
  Attributes:
    router: Router class
    rank: LoRA rank
    alpha = LoRA aplha
    lora_init_A: LoRA A initializer
    lora_init_B: LoRA B initializer
    lora_axis_names_A: Sharding axis names for LoRA A
    lora_axis_names_B: Sharding axis names for LoRA B
    dtype: Activation dtype
    output_dim: LoRA output dimensions
    num_heads: Number of heads
  """
  rank: int = 2
  lora_init_A: Initializer = nn.initializers.normal(stddev=2e-2)
  lora_init_B: Initializer = nn.initializers.zeros
  lora_axis_names_A: Sequence[str] = ('mlp', 'rank')
  lora_axis_names_B: Sequence[str] = ('rank', 'mlp')
  alpha = 16
  num_heads: int = 1
  output_dim: Optional[int] = None
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x: Array, **kwargs) -> Array:

    *rest, hidden = x.shape    

    x = jax.lax.convert_element_type(x, self.dtype)

    #[hidden, rank]
    lora_a = partitioning.param_with_axes(
              'lora_A',
              self.lora_init_A,
              (hidden, self.rank),
              jnp.float32,
              axes=self.lora_axis_names_A)
    
    lora_a = jax.lax.convert_element_type(lora_a, self.dtype)

    #[batch, seq_len, rank]
    ax = jnp.einsum('...d,dr->...r', 
                         x,
                         lora_a)                        

    # Add expert axis name to the partitioning axes
    ax = partitioning.with_sharding_constraint(ax, ('batch', 'length', 'unmodeled'))
    ax = jax.lax.convert_element_type(ax, self.dtype)

    #[rank, hidden]
    lora_b = partitioning.param_with_axes(
              'lora_B',
              self.lora_init_B,
              (self.rank, (self.output_dim if self.output_dim else hidden)),
              jnp.float32,
              axes=self.lora_axis_names_B)
    
    lora_b = jax.lax.convert_element_type(lora_b, self.dtype)
    
    #[batch, seq_len, rank]
    bax = jnp.einsum('...r,rd->...d', 
                         ax,
                         lora_b)  

    bax = bax * (self.alpha / self.rank)

    # Reshape to [batch, seq_len, num_heads, head_dim]
    bax = jnp.reshape(bax, (*rest, self.num_heads, hidden // self.num_heads))

    return bax