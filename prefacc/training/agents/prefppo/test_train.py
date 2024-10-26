"""PrefPPO tests."""
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import jax
from prefacc.training.agents.prefppo import train as prefppo


class PrefPPOTest(parameterized.TestCase):
    """Tests for PrefPPO module."""

    # def testTrain(self):
    #   """Test PrefPPO with a simple env."""
    #   fast = envs.get_environment('fast')
    #   _, _, metrics = prefppo.train(
    #       fast,
    #       num_timesteps=2**12,
    #       episode_length=128,
    #       num_envs=64,
    #       learning_rate=3e-4,
    #       entropy_cost=1e-2,
    #       discounting=0.95,
    #       unroll_length=5,
    #       batch_size=64,
    #       num_minibatches=8,
    #       num_updates_per_batch=4,
    #       normalize_observations=True,
    #       seed=2,
    #       num_evals=3,
    #       reward_scaling=10,
    #       normalize_advantage=False)
    #   self.assertGreater(metrics['eval/episode_reward'], 3)
    #   self.assertEqual(fast.reset_count, 2)  # type: ignore
    #   self.assertEqual(fast.step_count, 4)  # type: ignore

    # def testTrainV2(self):
    #   """Test PrefPPO with a v2 env."""
    #   _, _, _ = prefppo.train(
    #       envs.get_environment('inverted_pendulum', backend='spring'),
    #       num_timesteps=2**15,
    #       episode_length=1000,
    #       num_envs=64,
    #       learning_rate=3e-4,
    #       entropy_cost=1e-2,
    #       discounting=0.95,
    #       unroll_length=5,
    #       batch_size=64,
    #       num_minibatches=8,
    #       num_updates_per_batch=4,
    #       normalize_observations=True,
    #       seed=2,
    #       reward_scaling=10,
    #       normalize_advantage=False)

    # @parameterized.parameters(True, False)
    # def testNetworkEncoding(self, normalize_observations):
    #   env = envs.get_environment('fast')
    #   original_inference, params, _ = prefppo.train(
    #       env,
    #       num_timesteps=128,
    #       episode_length=128,
    #       num_envs=128,
    #       normalize_observations=normalize_observations)
    #   normalize_fn = lambda x, y: x
    #   if normalize_observations:
    #     normalize_fn = running_statistics.normalize
    #   ppo_network = ppo_networks.make_ppo_networks(env.observation_size,
    #                                                env.action_size, normalize_fn)
    #   inference = ppo_networks.make_inference_fn(ppo_network)
    #   byte_encoding = pickle.dumps(params)
    #   decoded_params = pickle.loads(byte_encoding)

    #   # Compute one action.
    #   state = env.reset(jax.random.PRNGKey(0))
    #   original_action = original_inference(decoded_params)(
    #       state.obs, jax.random.PRNGKey(0))[0]
    #   action = inference(decoded_params)(state.obs, jax.random.PRNGKey(0))[0]
    #   self.assertSequenceEqual(original_action, action)
    #   env.step(state, action)

    # def testTrainDomainRandomize(self):
    #   """Test PrefPPO with domain randomization."""

    #   def rand_fn(sys, rng):
    #     @jax.vmap
    #     def get_offset(rng):
    #       offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
    #       pos = sys.link.transform.pos.at[0].set(offset)
    #       return pos

    #     sys_v = sys.tree_replace({'link.inertia.transform.pos': get_offset(rng)})
    #     in_axes = jax.tree.map(lambda x: None, sys)
    #     in_axes = in_axes.tree_replace({'link.inertia.transform.pos': 0})
    #     return sys_v, in_axes

    #   _, _, _ = prefppo.train(
    #       envs.get_environment('inverted_pendulum', backend='spring'),
    #       num_timesteps=2**15,
    #       episode_length=1000,
    #       num_envs=64,
    #       learning_rate=3e-4,
    #       entropy_cost=1e-2,
    #       discounting=0.95,
    #       unroll_length=5,
    #       batch_size=64,
    #       num_minibatches=8,
    #       num_updates_per_batch=4,
    #       normalize_observations=True,
    #       seed=2,
    #       reward_scaling=10,
    #       normalize_advantage=False,
    #       randomization_fn=rand_fn,
    #   )

    # def testTrainBuffer(self):
    #   """Test PrefPPO with prefill."""
    #   fast = envs.get_environment('inverted_pendulum', backend='spring')
    #   _, _, metrics = prefppo.train(
    #       fast,
    #       num_timesteps=2**15,
    #       episode_length=128,
    #       num_envs=64,
    #       learning_rate=3e-4,
    #       entropy_cost=1e-2,
    #       discounting=0.95,
    #       unroll_length=5,
    #       batch_size=64,
    #       num_minibatches=8,
    #       num_updates_per_batch=4,
    #       normalize_observations=True,
    #       seed=2,
    #       num_evals=3,
    #       reward_scaling=10,
    #       normalize_advantage=False,
    #       min_replay_size=64,
    #       max_replay_size=128 * 64)
    #   self.assertGreater(metrics['eval/episode_reward'], 5)
    #   # self.assertEqual(fast.reset_count, 2)  # type: ignore
    # self.assertEqual(fast.step_count, 3)  # type: ignore

    # def testTrainAnt(self):
    #   env = envs.get_environment('ant', backend='positional')
    #   _, _, metrics = prefppo.train(
    #       env,
    #       num_timesteps=50_000_000, 
    #       num_evals=20, 
    #       reward_scaling=10, 
    #       episode_length=1000, 
    #       normalize_observations=True, 
    #       action_repeat=1, 
    #       unroll_length=5, 
    #       num_minibatches=32, 
    #       num_updates_per_batch=4, 
    #       discounting=0.97, 
    #       learning_rate=3e-4, 
    #       entropy_cost=1e-2, 
    #       num_envs=4096, 
    #       batch_size=2048, 
    #       seed=1, 
    #       num_prefs=2000
    #   )

    # def testTrainReacher(self):
    #   env = envs.get_environment('reacher', backend='positional')
    #   _, _, metrics = prefppo.train(
    #       env,
    #       num_timesteps=50_000_000,  # Increased timesteps
    #       num_evals=20,
    #       reward_scaling=5,  # Adjusted reward scaling
    #       episode_length=100,
    #       normalize_observations=True,
    #       action_repeat=4,
    #       unroll_length=50,
    #       num_minibatches=32,
    #       num_updates_per_batch=8,
    #       discounting=0.95,
    #       learning_rate=3e-4,  # Adjusted learning rate
    #       entropy_cost=1e-3,  # Increased exploration
    #       num_envs=2048,
    #       batch_size=256,
    #       seed=1,
    #       num_prefill_iterations = 20,
    #       num_rm_batches = 16,
    #       num_prefs=4000
    # )

#  def testTrainPusher(self):
#    env = envs.get_environment('ant', backend='positional')
#    _, _, metrics = prefppo.train(
#        env,
#        num_timesteps=8_000_000,  # Increased timesteps
#        num_evals=10,
#        reward_scaling=5,  # Adjusted reward scaling
#        episode_length=1000,
#        normalize_observations=True,
#        action_repeat=1,
#        unroll_length=5,
#        num_minibatches=32,
#        num_updates_per_batch=8,
#        discounting=0.95,
#        learning_rate=3e-4,  # Adjusted learning rate
#        entropy_cost=1e-2,  # Increased exploration
#        num_envs=4096,
#        batch_size=2048,
#        seed=3,
#        num_prefs=4000
#    )
#    self.assertGreater(metrics['eval/reward'], 1000)
#
    # _, _, metrics = prefppo.train(
    #     env,
    #     num_timesteps=50_000_000,
    #     num_evals=20,
    #     reward_scaling=5,
    #     episode_length=1000,
    #     normalize_observations=True,
    #     action_repeat=1,
    #     unroll_length=30,
    #     num_minibatches=16,
    #     num_updates_per_batch=8,
    #     discounting=0.95,
    #     learning_rate=3e-4,
    #     entropy_cost=1e-2,
    #     num_envs=2048,
    #     batch_size=512,
    #     seed=3,
    #     num_prefs=10000
    # )

    def testTrainAnt(self):
        env = envs.get_environment('ant', backend='positional')
        _, _, metrics = prefppo.train(
            env,
            num_timesteps=4_000_000,
            num_evals=20,
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
            num_envs=1024, #106.123s mit 1024, 512, 98.399s mit 2048 512
            batch_size=512,
            num_rm_batches=2,
            seed=1,
            num_prefs=2000)

        self.assertGreater(metrics['eval/episode_reward'], 1000)

#   def testTrainHalfCheetah(self):
#     env = envs.get_environment('halfcheetah', backend='positional')
#     _, _, metrics = prefppo.train(
#         env,
# num_timesteps=20_000_000, num_evals=10, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1
#   , num_prefs=700)
    # def testTrainPusher(self):
    #     env = envs.get_environment('pusher', backend='positional')
    #     _, _, metrics = prefppo.train(
    #     env,
    #     num_timesteps=50_000_000,
    #     num_evals=20,
    #     reward_scaling=5,
    #     episode_length=100,
    #     normalize_observations=True,
    #     action_repeat=1,
    #     unroll_length=30,
    #     num_minibatches=16,
    #     num_updates_per_batch=8,
    #     discounting=0.95,
    #     learning_rate=3e-4,
    #     entropy_cost=1e-2,
    #     num_envs=2048,
    #     batch_size=512,
    #     seed=3,
    #     num_prefill_iterations = 20,
    #     num_rm_batches = 16,
    #     num_prefs=4000)
    
    #     self.assertGreater(metrics['eval/episode_reward'], 0)  # Adjust this threshold as needed
    # def testTrainIDP(self):
    #     env = envs.get_environment('inverted_double_pendulum', backend='positional')
    #     _, _, metrics = prefppo.train(
    #     env,
    #     num_timesteps=20_000_000,
    #     num_evals=20,
    #     reward_scaling=10,
    #     episode_length=1000,
    #     normalize_observations=True,
    #     action_repeat=1,
    #     unroll_length=5,
    #     num_minibatches=32,
    #     num_updates_per_batch=4,
    #     discounting=0.97,
    #     learning_rate=3e-4,
    #     entropy_cost=1e-2,
    #     num_envs=2048,
    #     batch_size=1024,
    #     seed=1
    # )
    #     self.assertGreater(metrics['eval/episode_reward'], 0)  # Adjust this threshold as needed


if __name__ == '__main__':
    absltest.main()
