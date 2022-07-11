import carla

from common.carla_environment.action_sampler import CarlaPIDLongitudinalSampler


# Agent is any class that implement run_step function
# which return corresponding numpy array of action.

class LongitudinalAgent:
    def __init__(self, env):
        self.sampler = CarlaPIDLongitudinalSampler(env, max_step=float('inf'))

    def run_step(self):
        actions, _ = self.sampler.sample()

        return actions
