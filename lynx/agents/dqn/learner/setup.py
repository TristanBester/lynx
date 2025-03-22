import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from jumanji.env import Environment
from omegaconf import DictConfig

from lynx.agents.dqn.learner.learn import setup_learn_fn
from lynx.agents.dqn.learner.warmup import setup_warmup_fn
from lynx.networks.actor import Actor
from lynx.types.types import OffPolicyLearnerState, OnlineAndTarget, Transition


def setup_learner(env: Environment, key: chex.PRNGKey, config: DictConfig):
    # Setup the networks
    key, subkey = jax.random.split(key)
    q_network, eval_network, params = _setup_networks(env, subkey, config)

    # Setup the optimiser
    optim, optim_state = _setup_optimiser(params, config)

    # Setup the buffer
    buffer, buffer_state = _setup_buffer(env, config)

    # Setup the learn function
    learn_fn = setup_learn_fn(env, q_network.apply, optim.update, buffer, config)
    learn_fn = jax.pmap(learn_fn, axis_name="device")

    # Setup the warmup function
    warmup_fn = setup_warmup_fn(env, q_network.apply, params, buffer, config)
    warmup_fn = jax.pmap(warmup_fn, axis_name="device")

    # Reset the environment states and timesteps across devices and batches
    key, subkey = jax.random.split(key)
    env_states, timesteps = _reset_pmap(subkey, env, config)

    # Replicate across devices and batches
    params = _replicate_params_across_batches_and_devices(params, config)
    optim_state = _replicate_opt_state_across_batches_and_devices(optim_state, config)
    buffer_state = _replicate_buffer_state_across_batches_and_devices(
        buffer_state, config
    )

    # Warmup the buffer.
    key, subkey = jax.random.split(key)
    warmup_keys = _get_batch_device_keys(subkey, config)

    # breakpoint()

    env_states, timesteps, _, buffer_state = warmup_fn(
        env_states, timesteps, buffer_state, warmup_keys
    )

    # Initialise learner state.
    key, subkey = jax.random.split(key)
    step_keys = _get_batch_device_keys(subkey, config)
    init_learner_state = OffPolicyLearnerState(
        params=params,
        opt_state=optim_state,
        buffer_state=buffer_state,
        env_states=env_states,
        timesteps=timesteps,
        key=step_keys,  # type: ignore
    )
    return learn_fn, eval_network, init_learner_state


def _setup_networks(env: Environment, key: chex.PRNGKey, config: DictConfig):
    q_network_encoder = hydra.utils.instantiate(config.network.encoder)
    q_network_backbone = hydra.utils.instantiate(config.network.backbone)

    train_q_network_head = hydra.utils.instantiate(
        config.network.head,
        action_dim=env.action_spec.num_values,
        epsilon=config.train.training_epsilon,
    )
    eval_q_network_head = hydra.utils.instantiate(
        config.network.head,
        action_dim=env.action_spec.num_values,
        epsilon=0.0,
    )

    train_q_network = Actor(
        encoder=q_network_encoder,
        backbone=q_network_backbone,
        head=train_q_network_head,
    )
    eval_q_network = Actor(
        encoder=q_network_encoder,
        backbone=q_network_backbone,
        head=eval_q_network_head,
    )

    # Initialise parameters
    init_obs = env.observation_spec.generate_value()
    init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)

    online_params = train_q_network.init(key, init_obs_batch)
    target_params = online_params
    params = OnlineAndTarget(online=online_params, target=target_params)  # type: ignore
    return train_q_network, eval_q_network, params


def _setup_optimiser(params: OnlineAndTarget, config: DictConfig):
    optim = optax.chain(
        optax.clip_by_global_norm(config.train.max_grad_norm),
        optax.adam(config.train.learning_rate, eps=1e-5),
    )
    optim_state = optim.init(params.online)
    return optim, optim_state


def _setup_buffer(env: Environment, config: DictConfig):
    buffer = fbx.make_item_buffer(
        max_length=config.train.buffer_size,
        min_length=config.train.batch_size,
        sample_batch_size=config.train.batch_size,
        add_batches=True,
        add_sequences=True,
    )

    # Create a sample transition to initialise the buffer
    # TODO: Why do we add batch dimension and then remove it when adding to transition?
    init_obs = env.observation_spec.generate_value()
    init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
    transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_obs_batch),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_obs_batch),
        info={
            "episode_return": 0.0,
            "episode_length": 0,
            "is_terminal_step": False,
        },
    )
    buffer_state = buffer.init(transition)
    return buffer, buffer_state


def _reset_pmap(key, env, config):
    """Reset the environment states and timesteps across devices and batches.

    Reset the environment using VMAP to get enough states and timesteps.
    Reshape the states and timesteps to support pmap.
    """
    key, *env_keys = jax.random.split(
        key,
        jax.device_count()
        * config.train.updates_per_device
        * config.train.envs_per_device
        + 1,
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )

    def reshape_states(x: chex.Array) -> chex.Array:
        return x.reshape(
            (
                jax.device_count(),
                config.train.updates_per_device,
                config.train.envs_per_device,
            )
            + x.shape[1:]
        )

    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)
    return env_states, timesteps


def _replicate_params_across_batches_and_devices(params, config):
    """Two step replication process.

    This two-step approach allows the system to have identical copies of the
    parameters for each combination of batch and device, enabling parallel
    computation across both dimensions.
    """

    def broadcast(x: chex.Array) -> chex.Array:
        return jnp.broadcast_to(x, (config.train.updates_per_device,) + x.shape)

    batch_replicated_params = jax.tree_util.tree_map(broadcast, params)
    device_batch_replicated_params = flax.jax_utils.replicate(  # type: ignore
        batch_replicated_params, devices=jax.devices()
    )
    return device_batch_replicated_params


def _replicate_opt_state_across_batches_and_devices(opt_state, config):
    def broadcast(x: chex.Array) -> chex.Array:
        return jnp.broadcast_to(x, (config.train.updates_per_device,) + x.shape)

    batch_replicated_opt_state = jax.tree_util.tree_map(broadcast, opt_state)
    device_batch_replicated_opt_state = flax.jax_utils.replicate(  # type: ignore
        batch_replicated_opt_state, devices=jax.devices()
    )
    return device_batch_replicated_opt_state


def _replicate_buffer_state_across_batches_and_devices(buffer_state, config):
    def broadcast(x: chex.Array) -> chex.Array:
        return jnp.broadcast_to(x, (config.train.updates_per_device,) + x.shape)

    batch_replicated_buffer_state = jax.tree_util.tree_map(broadcast, buffer_state)
    device_batch_replicated_buffer_state = flax.jax_utils.replicate(  # type: ignore
        batch_replicated_buffer_state, devices=jax.devices()
    )
    return device_batch_replicated_buffer_state


def _get_batch_device_keys(key, config):
    def reshape_keys(x: chex.Array) -> chex.Array:
        return x.reshape(
            (jax.device_count(), config.train.updates_per_device) + x.shape[1:]
        )

    batch_device_keys = jax.random.split(
        key, jax.device_count() * config.train.updates_per_device
    )
    return reshape_keys(jnp.stack(batch_device_keys))
