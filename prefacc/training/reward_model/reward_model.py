"""Reward model definition and inference function."""

from typing import Callable, Sequence

from brax.training import types
from brax.training.networks import FeedForwardNetwork
from brax.training.networks import MLP
from flax import linen
import jax
import jax.numpy as jnp

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]


def make_reward_network(
    obs_size: int,
    action_size: int,
    preproceess_observation_fn:
    types.PreprocessObservationFn = types.
    identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu
) -> FeedForwardNetwork:
  reward_model_module = MLP(
    layer_sizes=list(hidden_layer_sizes) + [1],
    activation=activation,
    kernel_init=jax.nn.initializers.lecun_normal())
  
  def apply(processor_params, reward_model, params, obs, action):
    obs = preproceess_observation_fn(obs, processor_params)
    inputs = jnp.concatenate([obs, action], axis=-1)
    return jnp.squeeze(reward_model.apply(params, inputs), axis=-1)
  
  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  dummy_input = jnp.concatenate([dummy_obs, dummy_action], axis=-1)
  return FeedForwardNetwork(
    init=lambda key: reward_model_module.init(key, dummy_input), apply=apply)