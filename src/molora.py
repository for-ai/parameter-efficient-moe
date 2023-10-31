"""MoLoRa implementation"""

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning
from flaxformer.types import Array, Initializer, DType

from src import routing


class MoLoRa(nn.Module):
  """MoLoRa implementation
  
  Attributes:
    router: Router class
    rank: LoRA rank
    alpha = LoRA aplha
    lora_init_A: LoRA A initializer
    lora_init_B: LoRA B initializer
    lora_axis_names_A: Sharding axis names for LoRA A
    lora_axis_names_B: Sharding axis names for LoRA B
    num_experts: Number of expert
    dtype: Activation dtype
    output_dim: LoRA output dimensions
  """
  router: routing.Router
  rank: int = 2
  lora_init_A: Initializer = nn.initializers.normal(stddev=2e-2)
  lora_init_B: Initializer = nn.initializers.zeros
  lora_axis_names_A: Sequence[str] = ('unmodeled', 'mlp', 'unmodeled')
  lora_axis_names_B: Sequence[str] = ('unmodeled', 'unmodeled', 'mlp')
  alpha = 16
  num_experts: int = 1
  dtype: DType = jnp.float32
  output_dim: Optional[int] = None

  @nn.compact
  def __call__(self, x: Array, **kwargs) -> Array:

    *rest, hidden = x.shape    

    #x = jax.lax.convert_element_type(x, self.dtype)
    
    #[num_experts, hidden, rank]
    molora_a = partitioning.param_with_axes(
              'lora_A',
              self.lora_init_A,
              (self.num_experts, hidden, self.rank),
              jnp.float32,
              axes=self.lora_axis_names_A)
    
    molora_a = jax.lax.convert_element_type(molora_a, self.dtype)

    #[batch, seq_len, num_experts, rank]
    ax = jnp.einsum('bsd,edr->bser', 
                         x,
                         molora_a)                        

    # Add expert axis name to the partitioning axes
    ax = partitioning.with_sharding_constraint(ax, ('batch', 'length', 'expert', 'rank'))
    ax = jax.lax.convert_element_type(ax, self.dtype)

    #[num_experts, rank, output_dim]
    molora_b = partitioning.param_with_axes(
              'lora_B',
              self.lora_init_B,
              (self.num_experts, self.rank, (self.output_dim if self.output_dim else hidden)),
              jnp.float32,
              axes=self.lora_axis_names_B)
    
    molora_b = jax.lax.convert_element_type(molora_b, self.dtype)
    
    #[batch, seq_len, num_experts, rank]
    bax = jnp.einsum('bser,erd->bsed', 
                         ax,
                         molora_b)  
    
    bax = partitioning.with_sharding_constraint(bax, ('batch', 'length', 'expert') + tuple([self.lora_axis_names_B[-1]]))
    bax = jax.lax.convert_element_type(bax, self.dtype)
   
    #[batch, seq_len, num_experts]
    router_probs  = self.router(x, self.num_experts)
    router_probs = partitioning.with_sharding_constraint(router_probs, 
                                                         ('batch', 'length', 'expert'))

    #[batch, seq_len, hidden_dim]
    bax = jnp.einsum('...e,...ed->...d', 
                         router_probs, 
                         bax)

    return bax * (self.alpha / self.rank)


class MoLoRaAttention(nn.Module):
  """MoLoRa implementation for Attention class
  
  Attributes:
    router: Router class
    rank: LoRA rank
    alpha = LoRA aplha
    lora_init_A: LoRA A initializer
    lora_init_B: LoRA B initializer
    lora_axis_names_A: Sharding axis names for LoRA A
    lora_axis_names_B: Sharding axis names for LoRA B
    num_experts: Number of expert
    dtype: Activation dtype
    output_dim: LoRA output dimensions
    num_heads: Number of heads
  """
  router: routing.Router
  rank: int = 2
  lora_init_A: Initializer = nn.initializers.normal(stddev=2e-2)
  lora_init_B: Initializer = nn.initializers.zeros
  lora_axis_names_A: Sequence[str] = ('unmodeled', 'embed', 'unmodeled')
  lora_axis_names_B: Sequence[str] = ('unmodeled', 'unmodeled', 'joined_kv')
  alpha = 16
  num_experts: int = 1
  num_heads: int = 1
  dtype: DType = jnp.float32
  output_dim: Optional[int] = None

  @nn.compact
  def __call__(self, x: Array, **kwargs) -> Array:

    *rest, hidden = x.shape    

    #x = jax.lax.convert_element_type(x, self.dtype)
    
    #[num_experts, hidden, rank]
    molora_a = partitioning.param_with_axes(
              'lora_A',
              self.lora_init_A,
              (self.num_experts, hidden, self.rank),
              jnp.float32,
              axes=self.lora_axis_names_A)
    
    molora_a = jax.lax.convert_element_type(molora_a, self.dtype)

    #[batch, seq_len, num_experts, rank]
    ax = jnp.einsum('bsd,edr->bser', 
                         x,
                         molora_a)                        

    # Add expert axis name to the partitioning axes
    ax = partitioning.with_sharding_constraint(ax, ('batch', 'length', 'expert', 'rank'))
    ax = jax.lax.convert_element_type(ax, self.dtype)

    #[num_experts, rank, output_dim]
    molora_b = partitioning.param_with_axes(
              'lora_B',
              self.lora_init_B,
              (self.num_experts, self.rank, (self.output_dim if self.output_dim else hidden)),
              jnp.float32,
              axes=self.lora_axis_names_B)
    
    molora_b = jax.lax.convert_element_type(molora_b, self.dtype)
    
    #[batch, seq_len, num_experts, rank]
    bax = jnp.einsum('bser,erd->bsed', 
                         ax,
                         molora_b)  
    
    bax = partitioning.with_sharding_constraint(bax, ('batch', 'length', 'expert') + tuple([self.lora_axis_names_B[-1]]))
    bax = jax.lax.convert_element_type(bax, self.dtype)
   
    #[batch, seq_len, num_experts]
    router_probs = self.router(x, self.num_experts)
    router_probs = partitioning.with_sharding_constraint(router_probs, 
                                                         ('batch', 'length', 'expert'))

    #[batch, seq_len, hidden_dim]
    bax = jnp.einsum('...e,...ed->...d', 
                         router_probs, 
                         bax)
    
    # LoRA scaling
    bax = bax * (self.alpha / self.rank)

    # Reshape to [batch, seq_len, num_heads, head_dim]
    bax = jnp.reshape(bax, (*rest, self.num_heads, hidden // self.num_heads))
    
    return bax