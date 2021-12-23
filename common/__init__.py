from gym.envs.registration import register


register(
    id='Carla-v0',
    entry_point='common.environment.carla_environment:CarlaEnv',
)
