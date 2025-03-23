from typing import Any

import jax

from lynx.agents.dqn.learner.update.epoch import create_update_epoch_fn
from lynx.agents.dqn.learner.update.step import create_step_env_fn
from lynx.types.types import OffPolicyLearnerState


def create_update_fn(env, apply_fn, update_fn, buffer, config):
    def update(learner_state: OffPolicyLearnerState, _: Any):
        step_env = create_step_env_fn(env, apply_fn)
        # TODO: This should be called update step as it only performs on step
        # The epoch is in this function when we call jax.lax.scan
        update_epoch = create_update_epoch_fn(apply_fn, update_fn, buffer, config)

        # Rollout the policy in the environment
        learner_state, traj_batch = jax.lax.scan(
            step_env, learner_state, None, config.train.rollout_length
        )
        params, opt_state, buffer_state, key, env_states, timesteps = learner_state
        buffer_state = buffer.add(buffer_state, traj_batch)

        # Update the networks
        update_state = (params, opt_state, buffer_state, key)
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, None, config.train.updates_per_epoch
        )

        # Create the updated learner state
        params, opt_state, buffer_state, key = update_state
        learner_state = OffPolicyLearnerState(
            params=params,
            opt_state=opt_state,
            buffer_state=buffer_state,
            key=key,
            env_states=env_states,
            timesteps=timesteps,
        )
        return learner_state, (traj_batch.info, loss_info)

    return update
