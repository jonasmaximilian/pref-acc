"""pref-acc training types."""

from typing import NamedTuple

from brax.training.acme.types import NestedArray


class Transition(NamedTuple):
  """Container for a transition."""
  observation: NestedArray
  action: NestedArray
  reward: NestedArray
  reward_hat: NestedArray
  discount: NestedArray
  next_observation: NestedArray
  extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray