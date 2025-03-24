import jax
from jumanji.env import Environment

from lynx.agents.dqn.learner.update import create_update_fn
from lynx.types.types import OffPolicyLearnerState


def setup_learn_fn(env: Environment, apply_fn, update_fn, buffer, config):
    update_fn = create_update_fn(
        env=env,
        apply_fn=apply_fn,
        update_fn=update_fn,
        buffer=buffer,
        config=config,
    )

    def learn(learner_state: OffPolicyLearnerState):
        batch_update_fn = jax.vmap(update_fn, in_axes=(0, None), axis_name="batch")
        learner_state, (episode_statistics, training_statistics) = jax.lax.scan(
            batch_update_fn,
            learner_state,
            None,
            length=config.train.updates_per_epoch,
        )
        return learner_state, episode_statistics, training_statistics

    return learn
