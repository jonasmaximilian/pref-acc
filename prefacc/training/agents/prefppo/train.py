"""Preference-based proximal policy optimization (PrefPPO) training.

See: https://arxiv.org/abs/2111.03026
"""

import functools
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training.acting import Evaluator
from brax.training import gradients
from brax.training import pmap
from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
from prefacc.training import acting
from prefacc.training import types as prefacc_types
from prefacc.training.replay_buffers import RandomSamplingQueue
from prefacc.training.reward_model import losses as rm_losses
from prefacc.training.reward_model import reward_model as reward_model_networks


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = brax_types.Metrics

ReplayBufferState = Any
State = Union[envs.State, envs_v1.State]

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  policy_params: ppo_losses.PPONetworkParams
  reward_model_optimizer_state: optax.OptState
  reward_model_params: brax_types.Params
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jnp.ndarray


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak brax_types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return leaf.astype(leaf.dtype)
  return jax.tree_util.tree_map(f, tree)


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3, # 0.3
    gae_lambda: float = 0.95,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    segment_size: int = 50,
    num_prefs: int = 2000, # This is for testing atm, should be then used to calc num_prefs_per_training_step
    network_factory: brax_types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    reward_model_factory = reward_model_networks.make_reward_model_network,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    restore_checkpoint_path: Optional[str] = None,
):
  """PrefPPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    episode_length: the length of an environment episode
    action_repeat: the number of timesteps to repeat an action
    num_envs: the number of parallel environments to use for rollouts
      NOTE: `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    max_devices_per_host: maximum number of chips to use per host process
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy of
      the policy
    discounting: discounting rate
    seed: random seed
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    gae_lambda: General advantage estimation lambda
    deterministic_eval: whether to run the eval with a deterministic policy
    network_factory: function that generates networks for policy and value
      functions
    reward_model_factory: function that generates the reward model network
    progress_fn: a user-defined callback function for reporting/plotting metrics
    normalize_advantage: whether to normalize advantage estimate
    eval_env: an optional environment for eval only, defaults to `environment`
    policy_params_fn: a user-defined callback function that can be used for
      saving policy checkpoints
    randomization_fn: a user-defined callback function that generates randomized
      environments
    restore_checkpoint_path: the path used to restore previous model policy_params

  Returns:
    Tuple of (make_policy function, network policy_params, metrics)
  """
  assert batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)
  device_count = local_devices_to_use * process_count

  # Calculate min_replay_size as a fraction of num_timesteps
  min_replay_size = max(int(0.05 * num_timesteps), batch_size * unroll_length)
  
  # Ensure min_replay_size is at least batch_size * unroll_length
  min_replay_size = max(min_replay_size, batch_size * unroll_length)
  logging.info(f'Minimum replay size: {min_replay_size}')

  if min_replay_size >= num_timesteps:
    raise ValueError(
        'No training will happen because min_replay_size >= num_timesteps')
  
  # if max_replay_size is None:
  #   max_replay_size = num_timesteps

  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  logging.info(f'Number of prefill actor steps: {num_prefill_actor_steps}')
  num_prefs_per_epoch = num_prefs // num_evals 
  logging.info(f'Number of prefs per epoch: {num_prefs_per_epoch}')

  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat)
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  max_replay_size = max_replay_size or env_step_per_training_step
  logging.info('max_replay_size: %s', max_replay_size)

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, rb_key, key_env, eval_key = jax.random.split(local_key, 4)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value, key_reward_model = jax.random.split(global_key, 3)
  del global_key

  assert num_envs % device_count == 0

  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // local_device_count
    # all devices gets the same randomization rng
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )

  if isinstance(environment, envs.Env):
    wrap_for_training = envs.training.wrap
  else:
    wrap_for_training = envs_v1.wrappers.wrap_for_training

  env = wrap_for_training(
      environment,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )

  reset_fn = jax.jit(jax.vmap(env.reset))
  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(key_envs,
                         (local_devices_to_use, -1) + key_envs.shape[1:])
  env_state = reset_fn(key_envs)

  obs_size = env.observation_size
  action_size = env.action_size

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  ppo_network = network_factory(
      env_state.obs.shape[-1],
      env.action_size,
      preprocess_observations_fn=normalize)
  make_policy = ppo_networks.make_inference_fn(ppo_network)

  reward_model_network = reward_model_factory(
    env_state.obs.shape[-1],
    env.action_size)

  policy_optimizer = optax.adam(learning_rate=learning_rate)
  reward_model_optimizer = optax.adam(learning_rate=0.0005)

  dummy_obs = jnp.zeros((obs_size,))
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = prefacc_types.Transition(
      observation=dummy_obs,
      action=dummy_action,
      reward=0.,
      true_reward=0.,
      discount=0.,
      next_observation=dummy_obs,
      extras={
        'state_extras': {
            'truncation': 0.
        },
        'policy_extras': {
            'log_prob': 0.,
            'raw_action': jnp.zeros((action_size,))
        }
    })

  replay_buffer = RandomSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=segment_size) # This is the length of the segments

  policy_loss = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage)
  
  reward_model_loss = functools.partial(
      rm_losses.compute_reward_model_loss)

  policy_update = gradients.gradient_update_fn(
      policy_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
  
  reward_model_update = gradients.gradient_update_fn(
      reward_model_loss, reward_model_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)  
  

  def minibatch_step(
      carry, data: brax_types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState):
    policy_optimizer_state, policy_params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), policy_params, policy_optimizer_state = policy_update(
        policy_params,
        normalizer_params,
        data,
        key_loss,
        optimizer_state=policy_optimizer_state)

    return (policy_optimizer_state, policy_params, key), metrics

  def sgd_step(carry, unused_t, data: brax_types.Transition,
               normalizer_params: running_statistics.RunningStatisticsState):
    policy_optimizer_state, policy_params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (policy_optimizer_state, policy_params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (policy_optimizer_state, policy_params, key_grad),
        shuffled_data,
        length=num_minibatches)
    return (policy_optimizer_state, policy_params, key), metrics
  
  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params, reward_model_params: Params, env_state: Union[envs.State, envs_v1.State],
      buffer_state: ReplayBufferState, key: PRNGKey
  ) -> Tuple[running_statistics.RunningStatisticsState,
             Union[envs.State, envs_v1.State], ReplayBufferState]:
    policy = make_policy((normalizer_params, policy_params))
    reward_model = reward_model_networks.make_reward_model(reward_model_params, reward_model_network)
    env_state, transitions = acting.actor_step(
        env, env_state, policy, reward_model, key, extra_fields=('truncation',))
    
    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME)
    
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, env_state, buffer_state
  
  def unroll_and_insert(env: Union[envs.Env, envs_v1.Env, envs_v1.Wrapper],
                        env_state: State,
                        policy: brax_types.Policy,
                        reward_model,
                        buffer_state: ReplayBufferState,
                        key: PRNGKey,
                        unroll_length: int,
                        extra_fields: Sequence[str] = ()
                        ) -> Tuple[State, ReplayBufferState, prefacc_types.Transition]:
    
    @jax.jit
    def f(carry, unused_t):
      state, buffer_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      nstate, transition = acting.actor_step(
          env, state, policy, reward_model, current_key, extra_fields=extra_fields)
      buffer_state = replay_buffer.insert(buffer_state, transition)
      return (nstate, buffer_state, next_key), transition

    (final_state, buffer_state, _), data = jax.lax.scan(
        f, (env_state, buffer_state, key), (), length=unroll_length
    )
    return (final_state, buffer_state), data

  def training_step(
      carry: Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey],
      unused_t) -> Tuple[Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey], Metrics]:
    training_state, state, buffer_state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy(
        (training_state.normalizer_params, training_state.policy_params.policy))
    reward_model = reward_model_networks.make_reward_model(training_state.reward_model_params, reward_model_network)

    def f(carry, unused_t):
      current_state, buffer_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      (next_state, buffer_state), data = unroll_and_insert(
          env,
          current_state,
          policy,
          reward_model,
          buffer_state,
          current_key,
          unroll_length,
          extra_fields=('truncation',))

      return (next_state, buffer_state, next_key), data

    (state, buffer_state, key_generate_unroll), data = jax.lax.scan(
        f, (state, buffer_state, key_generate_unroll), (),
        length=batch_size * num_minibatches // num_envs)
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                  data)
    assert data.discount.shape[1:] == (unroll_length,)

    # Update normalization policy_params and normalize observations.
    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        data.observation,
        pmap_axis_name=_PMAP_AXIS_NAME)

    (policy_optimizer_state, policy_params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=data, normalizer_params=normalizer_params),
        (training_state.policy_optimizer_state, training_state.policy_params, key_sgd), (),
        length=num_updates_per_batch)

    new_training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        reward_model_optimizer_state=training_state.reward_model_optimizer_state,
        reward_model_params=training_state.reward_model_params,
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_step_per_training_step)
    return (new_training_state, state, buffer_state, new_key), metrics
  
  def prefill_replay_buffer(
      training_state: TrainingState, env_state: envs.State,
      buffer_state: ReplayBufferState, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, env_state, buffer_state = get_experience(
          training_state.normalizer_params, training_state.policy_params.policy,
          training_state.reward_model_params, env_state, buffer_state, key)
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          env_steps=training_state.env_steps + 1)
      return (new_training_state, env_state, buffer_state, new_key), ()
    
    return jax.lax.scan(
        f, (training_state, env_state, buffer_state, key), (),
        length=num_prefill_actor_steps)[0]
      

  prefill_replay_buffer = jax.pmap(
      prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

  def training_epoch(training_state: TrainingState, state: envs.State, buffer_state: ReplayBufferState,
                     key: PRNGKey) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    (training_state, state, buffer_state, _), loss_metrics = jax.lax.scan(
        training_step, (training_state, state, buffer_state, key), (),
        length=num_training_steps_per_epoch)
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, buffer_state, loss_metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State, buffer_state: ReplayBufferState,
      key: PRNGKey) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, buffer_state, key)
    training_state, env_state, buffer_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (num_training_steps_per_epoch *
           env_step_per_training_step *
           max(num_resets_per_eval, 1)) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, env_state, buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade
  
  def training_reward_model(
      training_state: TrainingState, buffer_state: ReplayBufferState
  ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
      
      def f(carry, unused):
          buffer_state = carry
          buffer_state, segment1 = replay_buffer.sample(buffer_state)
          buffer_state, segment2 = replay_buffer.sample(buffer_state)

          # idk if this works if grad_updates_per_step != 1
          segment1 = jax.tree_util.tree_map(
              lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
              segment1)
          segment2 = jax.tree_util.tree_map(
              lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
              segment2)

          summed_reward_s1 = jnp.sum(segment1.true_reward)
          summed_reward_s2 = jnp.sum(segment2.true_reward)
          pref = jax.lax.cond(summed_reward_s1 > summed_reward_s2, lambda _: [1., 0.], lambda _: [0., 1.], ())
          
          # pref_pair = prefacc_types.PreferencePair(segment1, segment2, label)
          pref_pair = prefacc_types.PreferencePair(segment1, segment2, pref)
          return (buffer_state), pref_pair

      # generate preference pairs
      (buffer_state), pref_data = jax.lax.scan(
          f, (buffer_state), (), length=num_prefs_per_epoch)
      
      # calc and apply loss

      rm_loss, rm_params, rm_optimizer_state = reward_model_update(training_state.reward_model_params, pref_data, optimizer_state=training_state.reward_model_optimizer_state)

      # loss, _ = rm_losses.compute_reward_model_loss(pref_data)
      # jax.debug.print("😭 {x} 😭", x=loss)

      logging.info("Done training rm!!!")

      new_training_state = training_state.replace(
          reward_model_params=rm_params,
          reward_model_optimizer_state=rm_optimizer_state)

      return (new_training_state, buffer_state), metrics
  
  training_reward_model = jax.pmap(training_reward_model,axis_name=_PMAP_AXIS_NAME)

  def calculate_pearson_correlation(training_state, buffer_state):
      buffer_state, batch = replay_buffer.sample(buffer_state)

      policy = make_policy((training_state.normalizer_params, training_state.policy_params.policy))
      reward_model = reward_model_networks.make_reward_model(training_state.reward_model_params, reward_model_network)

      correlation = jnp.corrcoef(batch.true_reward, batch.reward)[0, 1]
      return correlation
  
  calculate_pearson_correlation = jax.pmap(calculate_pearson_correlation, axis_name=_PMAP_AXIS_NAME)
  
  # Initialize policy params and training state.
  init_policy_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value),
  )

  init_reward_model_params = reward_model_network.init(key_reward_model)

  training_state = TrainingState(  # pytype: disable=wrong-arg-brax_types  # jax-ndarray
      policy_optimizer_state=policy_optimizer.init(init_policy_params),  # pytype: disable=wrong-arg-brax_types  # numpy-scalars
      policy_params=init_policy_params,
      reward_model_optimizer_state=reward_model_optimizer.init(init_reward_model_params),
      reward_model_params=init_reward_model_params,
      normalizer_params=running_statistics.init_state(
          specs.Array(env_state.obs.shape[-1:], jnp.dtype('float32'))),
      env_steps=0)

  if num_timesteps == 0:
    return (
        make_policy,
        (training_state.normalizer_params, training_state.policy_params),
        {},
    )

  if (
      restore_checkpoint_path is not None
      and epath.Path(restore_checkpoint_path).exists()
  ):
    logging.info('restoring from checkpoint %s', restore_checkpoint_path)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    target = training_state.normalizer_params, init_policy_params
    (normalizer_params, init_policy_params) = orbax_checkpointer.restore(
        restore_checkpoint_path, item=target
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params, policy_params=init_policy_params
    )

  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])
  
  # Replay buffer init
  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use))
  # buffer_state = replay_buffer.init(rb_key)
  # buffer_state = jax.device_put_replicated(
  #   buffer_state,
  #   jax.local_devices()[:local_devices_to_use])

  if not eval_env:
    eval_env = environment
  if randomization_fn is not None:
    v_randomization_fn = functools.partial(
        randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
    )
  eval_env = wrap_for_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )

  evaluator = Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.policy_params.policy)),
        training_metrics={})
    logging.info(metrics)
    progress_fn(0, metrics)

  num_prefill_iterations = 4  # Adjust this value as needed

  logging.info('enter prefill')
  for i in range(num_prefill_iterations):
    logging.info('prefill iteration %s', i)
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys)
    
    replay_size = jnp.sum(jax.vmap(
        replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info('replay size after prefill: %s', replay_size)
    
    if replay_size >= segment_size * 5:
      (training_state, buffer_state), metrics = training_reward_model(
          training_state, buffer_state)
      logging.info('Reward model training metrics: %s', metrics)
    else:
      logging.info('Skipping reward model training due to insufficient data')
    
    pearson_correlation = calculate_pearson_correlation(training_state, buffer_state)
    logging.info(f'Pearson correlation after prefill: {pearson_correlation}')
  pearson_correlation = calculate_pearson_correlation(training_state, buffer_state)
  logging.info(f'Pearson correlation after prefill: {pearson_correlation}')

  training_metrics = {}
  training_walltime = 0
  current_step = 0
  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      (training_state, env_state, buffer_state, training_metrics) = (
          training_epoch_with_timing(training_state, env_state, buffer_state, epoch_keys)
      )
      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s),
          in_axes=(0, None))(key_envs, key_envs.shape[1])
      env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

      # # Train reward model multiple times
      # for _ in range(1):  # Adjust the number of iterations as needed
      #     (training_state, buffer_state), _ = training_reward_model(
      #         training_state, buffer_state)
      #     logging.info('Reward model training done')

      pearson_correlation = calculate_pearson_correlation(training_state, buffer_state)
      logging.info(f'Pearson correlation between predicted and true rewards: {pearson_correlation}')
      training_metrics['pearson_correlation'] = pearson_correlation
      

    if process_id == 0:
      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params.policy)),
          training_metrics)
      logging.info(metrics)
      progress_fn(current_step, metrics)
      policy_params = _unpmap(
          (training_state.normalizer_params, training_state.policy_params)
      )
      policy_params_fn(current_step, make_policy, policy_params)

  total_steps = current_step
  assert total_steps >= num_timesteps

  replay_size = jnp.sum(jax.vmap(
      replay_buffer.size)(buffer_state)) * jax.process_count()
  
  logging.info('replay size at end: %s', replay_size)

  

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  policy_params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params.policy))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, policy_params, metrics)