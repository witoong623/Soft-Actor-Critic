import numpy as np

from agents.navigation.behavior_agent import BasicAgent
from common.carla_environment.action_sampler import CarlaPIDLongitudinalSampler, CarlaBiasActionSampler


# Agent is any class that implement run_step function
# which return corresponding numpy array of action.

class LongitudinalAgent:
    def __init__(self, env):
        self.sampler = CarlaPIDLongitudinalSampler(env, max_step=float('inf'))

    def run_step(self):
        actions, _ = self.sampler.sample()

        return actions


class CarlaBasicAgent:
    def __init__(self, env):
        self.agent = BasicAgent(env.ego, target_speed=env.desired_speed * 3.6)
        self.agent.set_global_plan(env.routeplanner.ait_route_planner.get_compatible_route_waypoint())
        self.agent.ignore_traffic_lights(active=True)
        self.agent.ignore_stop_signs(active=True)

    def run_step(self):
        vehicle_control = self.agent.run_step()

        acc = 0
        if vehicle_control.throttle > 0:
            acc = vehicle_control.throttle
        else:
            acc = -vehicle_control.brake

        return np.array([acc, vehicle_control.steer])


class ActionSamplerAgent:
    def __init__(self, env):
        self.sampler = CarlaBiasActionSampler(forward_only=False, use_brake=True)

    def run_step(self):
        actions, _ = self.sampler.sample()
        return actions
