"""MoV implementation"""

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning
from flaxformer.types import Array, Initializer, DType
from flaxformer.components import dense
from src import routing


class MoV(nn.Module):
  """MoV implementation
  
  Attributes:
    init: How to initialize the scaling variable.
    axis_name: The logical names of the variable axes, used for partitioning.
    dtype: The dtype of the activations for this module.
    router: Router class
    num_experts: Number of experts
  """
  router: routing.Router
  ia3_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.ones
  axis_name: Tuple[str] = ('unmodeled', 'mlp')
  num_experts: int = 1
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x, *args, **kwargs):
    del args 
    del kwargs
    *rest, hidden = x.shape
    scaling = partitioning.param_with_axes(
        'mov_scaling',
        self.ia3_init,
        (self.num_experts, hidden),
        jnp.float32,
        axes=self.axis_name
    )
    #[batch, seq_len, num_experts]
    router_probs  = self.router(x, self.num_experts)
    router_probs = partitioning.with_sharding_constraint(router_probs, 
                                                         ('batch', 'length', 'unmodeled'))                                                    

    #[num_experts, hidden_dim]
    scaling = jax.lax.convert_element_type(scaling, self.dtype)

    #[batch, seq_len, hidden_dim]
    scaling = jnp.einsum('...e,...ed->...d', 
                         router_probs, 
                         scaling)

    #[batch, seq_len, hidden_dim]
    #x = jax.lax.convert_element_type(x, self.dtype)
    return x * scaling


class MoVAttention(nn.Module):
  """MoV implementation for the Attention class.

  Attributes:
    init: How to initialize the scaling variable.
    axis_name: The logical names of the variable axes, used for partitioning.
    dtype: The dtype of the activations for this module.
    router: Router class
    num_experts: Number of experts
  """
  router: routing.Router
  ia3_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.ones
  axis_names: Tuple[str, str] = ('heads', 'unmodeled', 'kv')
  num_experts: int = 1
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x, *args, **kwargs):
    del args
    del kwargs
    *rest, heads, kv = x.shape
    scaling = partitioning.param_with_axes(
        'mov_scaling',
        self.ia3_init,
        (heads, self.num_experts, kv),
        jnp.float32,
        axes=self.axis_names
    )

    #[batch, seq_len, heads, kv_hidden]
    router_probs = self.router(x, self.num_experts)
    router_probs = partitioning.with_sharding_constraint(router_probs, 
                                                         ('batch', 'length', 'heads', 'unmodeled'))                                                           
    
    #[heads, num_experts, kv_hidden]
    scaling = jax.lax.convert_element_type(scaling, self.dtype)
    
    #[batch, seq_len, heads, kv_hidden]
    scaling = jnp.einsum('...e,...ed->...d', 
                         router_probs, 
                         scaling)
    #x = jax.lax.convert_element_type(x, self.dtype)
    return x * scaling