"Replay buffer for storing and sampling transitions"

from typing import Generic, Tuple

from brax.training.replay_buffers import QueueBase, ReplayBufferState, Sample
import jax
import jax.numpy as jnp


class RandomSamplingQueue(QueueBase[Sample], Generic[Sample]):
  """Implements a limited-size queue with random sampling.
  
  The difference to brax.UniformSamplingQueue is the sampling method
  Instead of choosing a random indices for every element in the batch,
  we choose a single random index and return the following sample_batch_size elements
  
  """

  def sample_internal(
      self, buffer_state: ReplayBufferState
  ) -> Tuple[ReplayBufferState, Sample]:
    if buffer_state.data.shape != self._data_shape:
      raise ValueError(
          f'Data shape expected by the replay buffer ({self._data_shape}) does '
          f'not match the shape of the buffer state ({buffer_state.data.shape}).')
    
    key, sample_key = jax.random.split(buffer_state.key)
    idx = jax.random.randint(
        sample_key,
        (self._sample_batch_size,),
        minval=buffer_state.sample_position,
        maxval=buffer_state.insert_position,
    )
    batch = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')
    return buffer_state.replace(key=key), self._unflatten_fn(batch)