import os
import carla
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from carla import ColorConverter as cc
from collections import deque
from queue import Queue
from common.utils import center_crop, convert_to_simplified_cityscape
from common.carla_environment.misc import get_pos, get_lane_dis_numba


class CarlaManualTrajectoryCollector:
    def __init__(self, world, carla_world, enable=False):
        self.world = world
        self.carla_world = carla_world
        self.vehicle = None
        self.enable = enable

        if not self.enable:
            return

        self.bp_library = self.carla_world.get_blueprint_library()

        self.obs_camera = None
        self.camera_trans = carla.Transform(carla.Location(x=0.88, z=1.675))
        self.camera_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
        self.camera_bp.set_attribute('image_size_x', '1280')
        self.camera_bp.set_attribute('image_size_y', '720')
        self.camera_bp.set_attribute('fov', '69')
        self.frame_data_queue = Queue()

        self.n_past_actions = 10
        self.action_queue = deque(maxlen=self.n_past_actions)
        self.desire_queue = deque(maxlen=self.n_past_actions)

        self.traveled_distance_diffs = deque(maxlen=100)
        self.previous_traveled_distance = 0

        self.desired_speed = 5.5
        self.out_lane_thres = 2.

        # For one step deley of this state.
        self.extra_state_queue = deque(maxlen=2)

        self.stored_transitions = []

        self.rewards_log = defaultdict(list)
        self.frame_number = 0

        self.prev_angle = None

    def setup_camera(self, vehicle):
        if not self.enable:
            return

        self._reset()

        self.vehicle = vehicle
        self.obs_camera = self.carla_world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.vehicle)
        self.obs_camera.listen(self.frame_data_queue.put)

        self.start_location = self.vehicle.get_location()

    def collect_transition(self, frame_number):
        if not self.enable:
            return

        self.frame_number = frame_number

        self._update_last_travel_distance(self.vehicle.get_location())

        obs = self._get_observation(frame_number)
        extra_stete = self._get_extra_state()
        action = self._get_action()
        reward = self._get_reward()
        done = self._get_terminal()

        self._save_transition(obs, extra_stete, action, reward, done)

        should_stop = self._get_should_stop()

        return done, should_stop

    def save_trajectory(self, file_path):
        with open(file_path, mode='wb') as f:
            pickle.dump(self.stored_transitions, f)

    def plot_reward_terms(self):
        for term, rewards in self.rewards_log.items():
            plt.plot(rewards)
            plt.title(term)
            plt.savefig(f'{term}-plot.jpeg')
            plt.clf()

    def _save_transition(self, obs, extra_state, action, reward, done):
        self.stored_transitions.append((obs, extra_state, action, reward, done))

    def _reset(self):
        self.action_queue.clear()
        self.stored_transitions.clear()
        self.traveled_distance_diffs.clear()

        for _ in range(self.n_past_actions):
            self.action_queue.append(np.array([0, 0], dtype=np.float16))

        self.extra_state_queue.append(np.concatenate(self.action_queue, dtype=np.float16))

    def _get_observation(self, frame_number):
        while True:
            data = self.frame_data_queue.get()
            self.frame_data_queue.task_done()
            if data.frame == frame_number:
                break

        data.convert(cc.CityScapesPalette)

        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]

        # BGR(OpenCV) > RGB
        raw_image = np.ascontiguousarray(array[:, :, ::-1])
        
        return self._transform_observation(raw_image)

    def _get_extra_state(self):
        action = self._get_action()

        self.action_queue.append(action)

        self.extra_state_queue.append(np.concatenate(self.action_queue, dtype=np.float16))

        state = self.extra_state_queue.popleft()

        return state

    def _get_action(self):
        carla_action = self.vehicle.get_control()
        throttle = carla_action.throttle
        brake = carla_action.brake
        steer = carla_action.steer

        acc = 0
        if throttle > 0.:
            acc = throttle
        elif brake > 0.:
            acc = -brake

        return np.array([acc, steer], dtype=np.float32)

    def _get_reward(self):
        # reward for speed tracking
        v = self.vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        r_speed = -abs(speed - self.desired_speed)
        
        # reward for collision
        r_collision = 0
        if self._does_vehicle_collide():
            r_collision = -1

        # reward for steering:
        carla_control = self.vehicle.get_control()
        r_steer = -carla_control.steer**2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.vehicle)
        self.current_lane_dis, w = get_lane_dis_numba(self.world.waypoints,
                                                      ego_x, ego_y,
                                                      direction_correction_multiplier=-1)
        r_out = 0
        if abs(self.current_lane_dis) > self.out_lane_thres:
            r_out = -100
        else:
            r_out = -abs(np.nan_to_num(self.current_lane_dis, posinf=self.out_lane_thres + 1, neginf=-(self.out_lane_thres + 1)))

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
            r_fast = -1

        # if it is faster than desired speed, minus the excess speed
        # and don't give reward from speed
        # r_fast *= lspeed_lon

        # cost for lateral acceleration
        r_lat = - abs(carla_control.steer) * lspeed_lon**2

        # cost for braking
        brake_cost = carla_control.brake * 2

        # cost for stopping
        r_stop = 0
        if self._does_vehicle_stop():
            r_stop = -1

        r = r_stop*200 + 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1 - brake_cost

        self._log_reward_by_term({
            'speed': lspeed_lon,
            'too_fast': 10*r_fast,
            'out_of_lane': 1*r_out,
            'steer': r_steer*5,
            'lat_acc': 0.2*r_lat,
            'brake': brake_cost,
        })

        return r

    def _get_terminal(self):
        if self._does_vehicle_collide():
            return True

        if abs(self.current_lane_dis) > self.out_lane_thres:
            return True

        return False

    def _transform_observation(self, image):
        cropped_size = (512, 1024)
        cropped_image = center_crop(image, cropped_size, shift_H=1.29)
        resized_obs = cv2.resize(cropped_image, (512, 256), interpolation=cv2.INTER_NEAREST)

        return convert_to_simplified_cityscape(resized_obs)

    def _does_vehicle_collide(self):
        return len(self.world.collision_sensor.history) > 0

    def _get_should_stop(self):
        return self.world.route_tracker.is_end_of_section

    def _does_vehicle_stop(self) -> bool:
        ''' vehicle stop if it doesn't move one meter in ten seconds window '''
        if len(self.traveled_distance_diffs) < self.traveled_distance_diffs.maxlen:
            return False

        return sum(self.traveled_distance_diffs) < 1.

    def _update_last_travel_distance(self, current_location):
        traveled_distance = self.start_location.distance(current_location)
        self.traveled_distance_diffs.append(abs(traveled_distance - self.previous_traveled_distance))
        self.previous_traveled_distance = traveled_distance

    def _log_reward_by_term(self, reward_terms):
        for term, reward in reward_terms.items():
            self.rewards_log[term].append(reward)


class SemanticSegmentationDatasetCollector:
    def __init__(self, world, carla_world, enable=False) -> None:
        self.world = world
        self.carla_world = carla_world
        self.vehicle = None
        self.enable = enable

        if not self.enable:
            return

        self.bp_library = self.carla_world.get_blueprint_library()

        self.seg_camera = None
        self.camera_trans = carla.Transform(carla.Location(x=0.88, z=1.675))

        self.camera_seg_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
        self.camera_seg_bp.set_attribute('image_size_x', '1280')
        self.camera_seg_bp.set_attribute('image_size_y', '720')
        self.camera_seg_bp.set_attribute('fov', '69')
        self.seg_frame_data_queue = Queue()

        self.rgb_camera = None
        self.camera_rgb_bp = self.bp_library.find('sensor.camera.rgb')
        self.camera_rgb_bp.set_attribute('image_size_x', '1280')
        self.camera_rgb_bp.set_attribute('image_size_y', '720')
        self.camera_rgb_bp.set_attribute('fov', '69')
        self.rgb_frame_data_queue = Queue()

        self.rgb_frame_dir = '/home/witoon/thesis/datasets/carla-ait/images'
        self.seg_frame_dir = '/home/witoon/thesis/datasets/carla-ait/labels'

    def setup_camera(self, vehicle):
        if not self.enable:
            return

        self.vehicle = vehicle

        self.seg_camera = self.carla_world.spawn_actor(self.camera_seg_bp, self.camera_trans, attach_to=self.vehicle)
        self.seg_camera.listen(self.seg_frame_data_queue.put)

        self.rgb_camera = self.carla_world.spawn_actor(self.camera_rgb_bp, self.camera_trans, attach_to=self.vehicle)
        self.rgb_camera.listen(self.rgb_frame_data_queue.put)

    def collect_transition(self, frame_number):
        rgb_frame = self._get_frame(frame_number, self.rgb_frame_data_queue, cc.Raw)
        seg_frame = self._get_frame(frame_number, self.seg_frame_data_queue, cc.CityScapesPalette)
        
        rgb_filepath = os.path.join(self.rgb_frame_dir, f'rgb_{frame_number}.png')
        seg_filepath = os.path.join(self.seg_frame_dir, f'seg_{frame_number}.png')

        self._save_frame(rgb_frame, rgb_filepath)
        self._save_frame(seg_frame, seg_filepath)

        return False, False

    def save_trajectory(self, filepath):
        pass

    def plot_reward_terms(self):
        pass

    def _save_frame(self, frame, filepath):
        cv2.imwrite(filepath, frame)

    def _get_frame(self, frame_number, queue_to_wait, frame_type):
        while True:
            data = queue_to_wait.get()
            queue_to_wait.task_done()
            if data.frame == frame_number:
                break

        data.convert(frame_type)

        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]

        # BGR(OpenCV) > RGB
        raw_image = np.ascontiguousarray(array[:, :, ::-1])
        
        return raw_image
