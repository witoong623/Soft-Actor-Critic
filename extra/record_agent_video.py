from main import get_config
from common.environment import initialize_environment
from common.carla_environment.custom_agent import LongitudinalAgent, CarlaBasicAgent, ActionSamplerAgent
from tqdm import tqdm


def fill_required_params(config):
    config.encoder_type = 'CNN'


if __name__ == '__main__':
    config = get_config()
    fill_required_params(config)
    initialize_environment(config)

    create_env_func = config.env_func
    create_env_kwargs = config.env_kwargs

    env = create_env_func(**create_env_kwargs)
    env.reset()

    # agent = LongitudinalAgent(env)
    agent = CarlaBasicAgent(env)
    # agent = ActionSamplerAgent(env)

    num_step = 0
    try:
        pbar = tqdm(total=config.max_episode_steps)
        while num_step < config.max_episode_steps:
            obs, reward, done, info = env.step(agent.run_step())

            if done:
                print('done')
                break

            pbar.update(1)
    finally:
        env.close()
        pbar.close()
