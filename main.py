from prefacc.training.agents.prefppo import train as prefppo

from brax import envs

env = envs.get_environment(env_name="ant", backend="positional")

prefppo.train(
        env,
        num_timesteps=2**15,
        episode_length=128,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        seed=2,
        num_evals=3,
        reward_scaling=10,
        normalize_advantage=False)
print("Training complete!")