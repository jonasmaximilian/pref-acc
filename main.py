from prefacc.training.agents.prefppo import train as prefppo

from brax import envs

env = envs.get_environment(env_name="ant", backend="positional")

# prefppo.train(
#         env,
#         num_timesteps=2**15,
#         episode_length=128,
#         num_envs=64,
#         learning_rate=3e-4,
#         entropy_cost=1e-2,
#         discounting=0.95,
#         unroll_length=5,
#         batch_size=64,
#         num_minibatches=8,
#         num_updates_per_batch=4,
#         normalize_observations=True,
#         seed=2,
#         num_evals=3,
#         reward_scaling=10,
#         normalize_advantage=False)

prefppo.train(
  env,
  num_timesteps=50_000_000, 
  num_evals=10, 
  reward_scaling=10, 
  episode_length=1000, 
  normalize_observations=True, 
  action_repeat=1, 
  unroll_length=5, 
  num_minibatches=32, 
  num_updates_per_batch=4, 
  discounting=0.97, 
  learning_rate=3e-4, 
  entropy_cost=1e-2, 
  num_envs=4096, 
  batch_size=2048, 
  seed=1,
  min_replay_size=2048,
  num_prefs=1400)
print("Training complete!")