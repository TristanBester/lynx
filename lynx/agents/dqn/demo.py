import hydra
import jax

from lynx.agents.dqn.learner.setup import setup_learner
from lynx.checkpoint import Checkpointer
from lynx.envs.factories.factory import make


@hydra.main(
    config_path="/Users/tristan/Projects/lynx/lynx/configs/base",
    config_name="dqn.yaml",
    version_base="1.2",
)
def main(config):
    # Create the checkpointer
    checkpointer = Checkpointer(
        model_name="dqn-snake",
        # checkpoint_dir=os.path.join(os.getcwd(), "checkpoints"),
        checkpoint_dir="/Users/tristan/Projects/lynx/checkpoints/seed-0",
        max_to_keep=6,
        keep_period=2,
    )
    #
    # checkpointer.upload_best_to_hf(
    #     repo_name="TristanBester/lynx",
    #     path_in_repo="checkpoints/dqn/puzzle",
    # )

    checkpointer.download_from_hf(
        repo_name="TristanBester/lynx",
        path_in_repo="checkpoints/dqn/puzzle",
    )

    params = checkpointer.restore()

    key = jax.random.PRNGKey(config.train.config.seed)

    # Setup the environments
    train_env, eval_env = make(config)

    state, timestep = train_env.reset(key)
    print(timestep.observation)

    # Create and initialse the learner
    key, subkey = jax.random.split(key)
    _, eval_network, _ = setup_learner(train_env, subkey, config)

    for _ in range(100):
        key, subkey = jax.random.split(key)
        state, timestep = eval_env.reset(subkey)
        eval_env.render(state)
        returns = 0

        while not timestep.last():
            action_dist = eval_network.apply(params, timestep.observation)
            # action = action_dist.mode()
            key, subkey = jax.random.split(key)
            action = action_dist.sample(seed=subkey)

            state, timestep = eval_env.step(state, action)
            eval_env.render(state)
            returns += timestep.reward
            print(returns)


if __name__ == "__main__":
    main()
