"""Router implementation."""

from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning as flax_partitioning
from flaxformer.types import Array, Initializer, DType
from flaxformer.components import dense

RouterOutput = Any

default_kernel_init = nn.initializers.normal(stddev=2e-2)
default_bias_init = nn.initializers.zeros

class RouterWeights(nn.Module):
  """Router module converting token inputs to router logits.

  Attributes:
    use_bias: Whether or not to use the bias term in computing the logits.
    dtype: Numerical float type for router logit computation.
    kernel_init: Initialization scheme for kernel.
    bias_init: Initialization scheme for bias.
    precision: XLA precision for array computations.
    axis: Axes along which to apply the dense router weights transformation.
      Defaults to final axis (typically the "hidden dimension").
    kernel_axis_names: Logical axis names to use for kernel sharding.
    reshape_kernel: Whether to reshape the kernel parameter to 2D for Adafactor.
  """
  use_bias: bool = True
  dtype: DType = jnp.bfloat16
  kernel_init: Initializer = default_kernel_init  # pytype: disable=annotation-type-mismatch  # jax-types
  bias_init: Initializer = default_bias_init
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  axis: Union[Iterable[int], int] = -1
  kernel_axis_names: Sequence[str] = ('mlp', 'unmodeled')
  reshape_kernel: bool = True

  @nn.compact
  def __call__(self, token_inputs: Array, num_experts: int) -> Array:
    """Applies RouterWeights module.

    Args:
      token_inputs: Flattened batch of tokens with shape <float>[num_groups,
        group_size, hidden_dim].
      num_experts: Number of experts.

    Returns:
      Router logits with shape <float>[num_groups, group_size, num_experts].
    """
    return dense.DenseGeneral(
        features=num_experts,
        axis=self.axis,
        use_bias=self.use_bias,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
        kernel_axis_names=self.kernel_axis_names,
        reshape_kernel=self.reshape_kernel,
        name='v')(
            token_inputs)


class Router(nn.Module):
  """Abstract base router class, defining router API and inner workings.

  Attributes:
    router_weights: Configurable module used to compute router logits from token
      inputs.
    jitter_noise: Amplitude of jitter noise applied to router logits.
    dtype: Numeric float type for returned combine array. All actual
      computations are performed in float32 of the input for stability.
    ignore_padding_tokens: Whether to ignore padding tokens during routing. Note
      that some routers (e.g. TokensChooseMaskedRouter) will completely ignore
      padding tokens, while others (e.g. TokensChooseScatterRouter and
      ExpertsChooseMaskedRouter) will simply down-weight the probability of
      selecting padding tokens.
  """
  router_weights: RouterWeights
  jitter_noise: float
  dtype: jnp.dtype
  ignore_padding_tokens: bool = True
  input_axis_names: Sequence[str] = ('batch', 'length', 'mlp')
  top_k: int = None
  load_balancing_loss: bool = False

  def __call__(self,
               token_inputs: Array,
               num_experts: int,
               apply_jitter: bool = True) -> RouterOutput:
    """Computes dispatch and combine arrays for routing to experts.

    Args:
      token_inputs: <float>[batch, seq_len, hidden_dim] inputs to
        send to experts.
      num_experts: Number of experts.
      apply_jitter: If true, apply jitter noise during routing.

    Returns:
      Router indices or mask arrays (depending on router type).
    """
    token_inputs = flax_partitioning.with_sharding_constraint(token_inputs, 
                                                              self.input_axis_names)

    router_probs, router_logits = self._compute_router_probabilities(
        token_inputs, num_experts, apply_jitter)

    if self.ignore_padding_tokens:
      # To identify non-padding tokens, we rely on the fact that padding tokens
      # in the inputs have already been masked in the default T5 architecture.
      # See
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L315
      # and
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L603.
      padding_mask = jnp.array((jnp.sum(jnp.abs(token_inputs), axis=-1) > 0),
                               dtype=token_inputs.dtype)
      router_logits *= jnp.expand_dims(padding_mask, axis=-1)
    else:
      padding_mask = None

    return router_probs

  def _compute_router_probabilities(self, token_inputs: Array, num_experts: int,
                                    apply_jitter: bool) -> Tuple[Array, Array]:
    """Computes router probabilities from input tokens.

    Args:
      token_inputs: <float>[batch, seq_len, hidden_dim] from which
        router probabilities are computed.
      num_experts: Number of experts.
      apply_jitter: If true, apply jitter noise.

    Returns:
      - <float32>[batch, seq_len, num_experts] probabilities for
        each token and expert. Used for routing tokens to experts.
      - <float>[batch, seq_len, num_experts] raw router logits.
        Used for computing router z-loss.
    """
    # For remainder of routing computation we use float32 to ensure stability.
    # See the discussion of "selective precision" in
    # https://arxiv.org/abs/2101.03961.
    token_inputs = jax.lax.convert_element_type(token_inputs, jnp.float32)

    if apply_jitter and self.jitter_noise > 0:
      token_inputs *= jax.random.uniform(
          self.make_rng('jitter'),
          token_inputs.shape,
          token_inputs.dtype,
          minval=1.0 - self.jitter_noise,
          maxval=1.0 + self.jitter_noise)

    # Shape: [batch, seq_len, num_experts]
    router_logits = self.router_weights(token_inputs, num_experts)

    router_probabilities = jax.nn.softmax(router_logits, axis=-1)
  
    if self.top_k is not None:
      topk_mask, top_k_indices = _top_k_mask(router_probabilities, self.top_k)
      router_axis_name = self.input_axis_names[:-1] + ('unmodeled',)
      topk_mask = flax_partitioning.with_sharding_constraint(topk_mask, router_axis_name)

    return router_probabilities * topk_mask if self.top_k is not None else router_probabilities, router_logits
  
  def _compute_routing_instructions(self, router_probs: Array,
                                    padding_mask: Optional[Array],
                                    expert_capacity: int) -> RouterOutput:
    """Computes instructions for routing inputs to experts."""
    raise NotImplementedError(
        'Router is an abstract class that should be subclassed.')


def _load_balancing_loss(router_probs: Array, expert_mask: Array = None) -> float:
  """Compute load balancing loss."""
  num_experts = router_probs.shape[-1]

  router_prob_per_expert = jnp.mean(
      router_probs, dtype=jnp.float32, axis=-2)

  if expert_mask is not None:
    tokens_per_expert = jnp.mean(
      expert_mask, dtype=jnp.float32, axis=-2)
    return jnp.mean(
      tokens_per_expert * router_prob_per_expert,
      dtype=jnp.float32) * num_experts**2
  else:
    return jnp.mean(
      router_prob_per_expert,
      dtype=jnp.float32) * num_experts**2


def _router_z_loss(router_logits: Array) -> float:
  """Compute router z-loss.

   The router z-loss was introduced in Designing Effective Sparse Expert Models
   (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
   small in an effort to improve stability.

  Args:
    router_logits: <float>[num_groups, tokens_per_group, num_experts] router
      logits.

  Returns:
    Scalar router z-loss.
  """
  num_groups, tokens_per_group, _ = router_logits.shape
  log_z = jax.nn.logsumexp(router_logits, axis=-1)
  z_loss = log_z**2
  return jnp.sum(z_loss, dtype=jnp.float32) / (num_groups * tokens_per_group)


def _favor_one_hot_slices() -> bool:
  """Returns true iff running on TPUs."""
  return jax.default_backend() == 'tpu' or jax.devices()[0].platform == 'tpu'


def _take_along_axis(array: Array, indices: Array, axis: int) -> Array:
  """Takes values from the input array by matching 1D index and data slices.

  This function serves the same purpose as jax.numpy.take_along_axis, except
  that it uses one-hot matrix multiplications under the hood on TPUs:
  (1) On TPUs, we use one-hot matrix multiplications to select elements from the
      array; this is particularly helpful for avoiding erroneous all-gather ops
      when running under pjit.
  (2) Otherwise, we fall back to jax.numpy.take_along_axis.

  Notes:
    - To simplify matters in case (1), we only support slices along the second
      or last dimensions.
    - We may wish to revisit (1) for very large arrays.

  Args:
    array: Source array.
    indices: Indices to take along each 1D slice of array.
    axis: Axis along which to take 1D slices.

  Returns:
    The indexed result.
  """
  if array.ndim != indices.ndim:
    raise ValueError(
        'indices and array must have the same number of dimensions; '
        f'{indices.ndim} vs. {array.ndim}.')

  if (axis != -1 and axis != array.ndim - 1 and  # Not last dimension
      axis != 1 and axis != -array.ndim + 1):  # Not second dimension
    raise ValueError(
        'Only slices along the second or last dimension are supported; '
        f'array.ndim = {array.ndim}, while axis = {axis}.')

  if _favor_one_hot_slices():
    one_hot_length = array.shape[axis]
    one_hot_indices = jax.nn.one_hot(indices, one_hot_length, axis=axis)

    if axis == -1 or array.ndim == 1:
      # Take i elements from last dimension (s).
      # We must use HIGHEST precision to accurately reproduce indexing
      # operations with matrix multiplications.
      result = jnp.einsum(
          '...s,...is->...i',
          array,
          one_hot_indices,
          precision=jax.lax.Precision.HIGHEST)
    else:
      # Take i elements from second dimension (s). We assume here that we always
      # want to slice along the second dimension.
      # We must use HIGHEST precision to accurately reproduce indexing
      # operations with matrix multiplications.
      result = jnp.einsum(
          'ns...,nis...->ni...',
          array,
          one_hot_indices,
          precision=jax.lax.Precision.HIGHEST)
    return jax.lax.convert_element_type(result, array.dtype)
  else:
    return jnp.take_along_axis(array, indices, axis=axis)

  
def _top_k_mask(array: Array, k: int) -> Tuple[Array, Array]:
  top_k_indices = jax.lax.top_k(array, k)[-1]
  mask = jax.nn.one_hot(top_k_indices, array.shape[-1], dtype=jnp.float32)
  mask = jnp.sum(mask, axis=-2)
  return mask, top_k_indices