import random
import numpy as np

from agents.navigation.behavior_agent import BehaviorAgent, BasicAgent
from agents.navigation.controller import PIDLongitudinalController


class CarlaPerfectActionSampler:
    def __init__(self, env) -> None:
        self.agent = BasicAgent(env.ego,
                                target_speed=env.desired_speed * 3.6)

        self.agent.set_global_plan(env.route_tracker.route_planner.get_carla_agent_compatible_route_waypoint())
        self.agent.ignore_traffic_lights(active=True)
        self.agent.ignore_stop_signs(active=True)

        self.time_step = 0
        self.max_time_step = 15 * env.frame_per_second

    def sample(self):
        self.time_step += 1
        action_command = self.agent.run_step()
        action = self._carla_command_to_action(action_command)

        return action, self.time_step == self.max_time_step

    def _carla_command_to_action(self, command):
        ''' Convert CARLA control command to environment action '''
        throttle = command.throttle
        brake = command.brake
        steer = command.steer

        if throttle > 0:
            acc = throttle
        elif brake > 0:
            acc = -brake
        else:
            acc = 0

        return np.array([acc, steer])


class CarlaPIDLongitudinalSampler:
    ''' Sample only longitudinal control '''
    def __init__(self, env, max_step=float('inf')):
        self.pid_controller = PIDLongitudinalController(env.ego, 1, 0.05, 0, dt=env.dt)
        self.target_speed = env.desired_speed * 3.6
        self.max_step = max_step
        self.count = 0

    def sample(self):
        long_control = self.pid_controller.run_step(self.target_speed)

        self.count += 1

        return np.array([long_control, 0], dtype=np.float32), self.count > self.max_step


class CarlaBiasActionSampler:
    def __init__(self, *args, forward_only=False, use_brake=False, 
                 max_step=float('inf'), try_correction=False, **kwargs) -> None:
        self.previous_action = None
        self.use_brake = use_brake
        self.forward_only = forward_only
        self.max_step = max_step
        self.try_collection = try_correction

        self.count = 0

    def sample(self):
        if self._should_use_previous_action():
            if self.try_collection:
                if self.previous_action[1] != 0:
                    action = np.array([self.previous_action[0], self.previous_action[1] * -1], dtype=np.float32)
                else:
                    action = self.previous_action
            else:
                action = self.previous_action
        else:
            action = self._sample_new_action()

        self.previous_action = action

        self.count += 1

        return action, self.count > self.max_step

    def _sample_new_action(self):
        forward_threshold = 0.4
        if not self.use_brake:
            forward_threshold = 0.0

        forward_prob = random.random()
        if forward_prob > forward_threshold:
            acc = self._sample_acc()
        else:
            acc = self._sample_brake()

        lateral_threshold  = 0.5
        if self.forward_only:
            lateral_threshold = 1.0

        lateral_prob = random.random()
        if lateral_prob > lateral_threshold:
            steer = self._sample_steer()
        else:
            steer = 0

        return np.array([acc, steer], dtype=np.float32)

    def _sample_acc(self):
        return max(40, random.random())

    def _sample_brake(self):
        return -max(0.05, random.random())

    def _sample_steer(self):
        return random.gauss(0, 1)

    def _should_use_previous_action(self):
        if self.previous_action is not None and self.previous_action[0] > 0:
            if np.random.randint(3) % 3:
                return True

        return False
