"""Reward model tests."""

from absl.testing import absltest
from prefacc.training.reward_model import reward_model


class RewardModelTest(absltest.TestCase):
  """Tests for reward model module."""

  def testRewardModelCreation(self):
    """Test reward model creation."""
    obs_size = 3
    action_size = 2
    model = reward_model.make_reward_model_network(obs_size, action_size)
    self.assertIsNotNone(model)


if __name__ == '__main__':
  absltest.main()