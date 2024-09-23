""" Reward model training losses

See: https://arxiv.org/pdf/1706.03741
"""

from brax.training.types import Metrics, Params
import jax
import jax.numpy as jnp
from prefacc.training.types import PreferencePair
from prefacc.training.agents.prefppo import reward_model as reward_model_networks


def compute_reward_model_loss(
    reward_model_params: Params,
    pref_pairs: PreferencePair,
    reward_model_network: reward_model_networks.RewardModelNetworks
) -> jnp.ndarray:
    reward_model_apply = reward_model_network.reward_model_network.apply

    def f(carry, pref_pair):
        _ = carry
        reward_hat_s1 = jnp.sum(reward_model_apply(reward_model_params, pref_pair.segment1.observation, pref_pair.segment1.action))
        reward_hat_s2 = jnp.sum(reward_model_apply(reward_model_params, pref_pair.segment2.observation, pref_pair.segment2.action))

        # probability that s1 is preferred
        prob_s1 = -jnp.log(1 + jnp.exp(reward_hat_s2 - reward_hat_s1))
        # probability that s2 is preferred
        prob_s2 = -jnp.log(1 + jnp.exp(reward_hat_s1 - reward_hat_s2))
        # This can be made even more numerically stable. See https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        loss = pref_pair.preference[0] * prob_s1 + pref_pair.preference[1] * prob_s2

        return _, loss

    # scan over pref_pairs
    (), losses = jax.lax.scan(
          f, (), pref_pairs)
    
    final_loss = -jnp.sum(losses)
    return final_loss

# def compute_reward_model_loss(
#     reward_model_params: Params,
#     pref_pairs: PreferencePair,
#     reward_model_network: reward_model_networks.RewardModelNetworks
# ) -> jnp.ndarray:
#     reward_model_apply = reward_model_network.reward_model_network.apply

#     def f(carry, pref_pair):
#         _ = carry
#         reward_hat_s1 = jnp.sum(reward_model_apply(reward_model_params, pref_pair.segment1.observation, pref_pair.segment1.action))
#         reward_hat_s2 = jnp.sum(reward_model_apply(reward_model_params, pref_pair.segment2.observation, pref_pair.segment2.action))

#         # Using sigmoid for numerical stability
#         # Clip the difference to avoid overflow in exp
#         delta_s1_s2 = jnp.clip(reward_hat_s2 - reward_hat_s1, -10, 10)
#         delta_s2_s1 = jnp.clip(reward_hat_s1 - reward_hat_s2, -10, 10)

#         # probability that s1 is preferred
#         prob_s1 = 1 / (1 + jnp.exp(delta_s1_s2))
#         # probability that s2 is preferred
#         prob_s2 = 1 / (1 + jnp.exp(delta_s2_s1))
#         # Loss calculation using sigmoid probabilities
#         loss = pref_pair.preference[0] * -jnp.log(prob_s1) + pref_pair.preference[1] * -jnp.log(prob_s2)

#         return _, loss

#     # scan over pref_pairs
#     (), losses = jax.lax.scan(
#           f, (), pref_pairs)
    
#     final_loss = jnp.sum(losses)  # Note the change from -jnp.sum to jnp.sum due to the change in loss calculation
#     return final_loss