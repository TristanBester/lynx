import hydra
import jax

from lynx.envs.factories.factory import make


@hydra.main(
    config_path="/Users/tristan/Projects/lynx/lynx/configs",
    config_name="dqn.yaml",
    version_base="1.2",
)
def main(config):
    # Create the checkpointer
    # checkpointer = Checkpointer(
    #     model_name="dqn-snake",
    #     # checkpoint_dir=os.path.join(os.getcwd(), "checkpoints"),
    #     checkpoint_dir="/Users/tristan/Projects/lynx/checkpoints",
    #     max_to_keep=6,
    #     keep_period=2,
    # )
    #
    # # checkpointer.upload_best_to_hf(
    # #     repo_name="TristanBester/lynx",
    # #     path_in_repo="checkpoints/dqn/snake",
    # # )
    #
    # checkpointer.download_from_hf(
    #     repo_name="TristanBester/lynx",
    #     path_in_repo="checkpoints/dqn/snake",
    # )

    # params = checkpointer.restore()

    key = jax.random.PRNGKey(config.experiment.seed)

    # Setup the environments
    train_env, eval_env = make(config)

    state, timestep = train_env.reset(key)
    print(timestep.observation)

    # # Create and initialse the learner
    # key, subkey = jax.random.split(key)
    # learn_fn, eval_network, learner_state = setup_learner(train_env, subkey, config)

    # for _ in range(100):
    #     key, subkey = jax.random.split(key)
    #     state, timestep = eval_env.reset(subkey)
    #     eval_env.render(state)

    #     while not timestep.last():
    #         action_dist = eval_network.apply(params, timestep.observation)
    #         action = action_dist.mode()

    #         state, timestep = eval_env.step(state, action)
    #         eval_env.render(state)


if __name__ == "__main__":
    main()
