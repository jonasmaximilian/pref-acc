""" Reward model training losses

See: https://arxiv.org/pdf/1706.03741
"""

from typing import Tuple

from brax.training.types import Metrics, Params
import jax
import jax.numpy as jnp
from prefacc.training.types import PreferencePair


def compute_reward_model_loss(
    reward_model_params: Params,
    pref_pairs: PreferencePair,
) -> jnp.ndarray:
# ) -> Tuple[jnp.ndarray, Metrics]:

    # calculate loss for a single pair
    def f(carry, pref_pair):
        _ = carry
        reward_hat_s1 = jnp.sum(pref_pair.segment1.reward)
        reward_hat_s2 = jnp.sum(pref_pair.segment2.reward)

        # probability that s1 is preferred over s2 (bradley terry model)
        prob_s1 = jnp.exp(reward_hat_s1) // jnp.exp(reward_hat_s1) + jnp.exp(reward_hat_s2)
        # probability that s2 is preferred
        prob_s2 = jnp.exp(reward_hat_s2) // jnp.exp(reward_hat_s1) + jnp.exp(reward_hat_s2)
        loss = pref_pair.preference[0] * jnp.log(prob_s1) + pref_pair.preference[1] * jnp.log(prob_s2)
        # jax.debug.print("🤯 {x} 🤯", x=loss)

        return _, loss

    # scan over pref_pairs
    (), losses = jax.lax.scan(
          f, (), pref_pairs)
    
    final_loss = -jnp.sum(losses)
    # jax.debug.print("🤯 {x} 🤯", x=final_loss)

    # return final_loss, {"reward_model_loss": final_loss}
    return final_loss