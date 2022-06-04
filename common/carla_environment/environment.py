import carla
import cv2
import gym
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agents.navigation.behavior_agent import BehaviorAgent, BasicAgent
from carla import ColorConverter as cc
from collections import deque
from copy import deepcopy
from enum import Enum, auto
from gym import spaces
from PIL import Image
from queue import Queue
from tqdm import trange

from .misc import get_pos, get_lane_dis_numba, get_vehicle_angle
from .route_planner import RoutePlanner
from .manual_route_planner import ManualRoutePlanner, TOWN7_PLAN, TOWN7_REVERSE_PLAN
from ..utils import center_crop, normalize_image

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.global_route_planner import RoadOption


class RouteMode(Enum):
    BASIC_RANDOM = auto()
    MANUAL_LAP = auto()


_walker_spawn_points_cache = []
_load_world = False


class CarlaEnv(gym.Env):
    start_wp_idx = 0

    def __init__(self, **kwargs):
        global _load_world
        self.host = 'localhost'
        self.port = 2000

        self.n_images = kwargs.get('n_frames', 1)
        observation_size = kwargs.get('image_size', [256, 512])
        self.obs_width = observation_size[1]
        self.obs_height = observation_size[0]
        self.obs_dtype = np.float16

        self.map = 'Town07'
        self.fps_mode = kwargs.get('fps_mode')
        if self.fps_mode == 'high':
            self.dt = 0.1
        else:
            self.dt = 0.2
        self.frame_per_second = round(1 / self.dt)
        self.reload_world = True
        self.use_semantic_camera = True

        camera_size = kwargs.get('camera_size')
        self.camera_width = camera_size[1]
        self.camera_height = camera_size[0]
        if kwargs.get('camera_fov'):
            self.camera_fov = kwargs.get('camera_fov')

        self.number_of_walkers = 0
        self.number_of_vehicles = 0
        self.number_of_wheels = [4]
        self.max_ego_spawn_times = 100
        self.max_time_episode = kwargs.get('max_episode_steps', 5000)
        self.max_waypt = 12
        # in m/s. 5.5 is 20KMH
        self.desired_speed = 5.5
        self.out_lane_thres = 2.

        # random or roundabout
        self.task_mode = 'random'

        self.dests = None

        # action and observation spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.obs_height, self.obs_width, 3), dtype=np.uint8)
        # steering, accel/brake
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)

        self.dry_run = kwargs.get('dry_run_init_env', False)
        if self.dry_run:
            print('dry run, exit init')
            return

        print('connecting to Carla server...')
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)
        if _load_world:
            self.world = self.client.get_world()
        else:
            self.world = self.client.load_world(self.map)
            _load_world = True
        print('Carla server connected!')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        if self.fps_mode == 'low':
            self.settings.max_substep_delta_time = 0.01666
            self.settings.max_substeps = 13
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.bp_library = self.world.get_blueprint_library()

        # spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # top left spawn point of town4
        self.lap_spwan_point_wp = self.world.get_map().get_waypoint(self.vehicle_spawn_points[1].location)

        self.walker_spawn_points = []
        # if we can cache more than 70% of spawn points then use cache
        if len(_walker_spawn_points_cache) > self.number_of_walkers * 0.7:
            def loc_to_transform(loc):
                x, y, z = loc
                loc = carla.Location(x=x, y=y, z=z)
                return carla.Transform(location=loc)

            self.walker_spawn_points = list(map(loc_to_transform, _walker_spawn_points_cache))
            print('load walker spwan points from cache')
        else:
            _walker_spawn_points_cache.clear()
            for i in range(self.number_of_walkers):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point.location = loc
                    self.walker_spawn_points.append(spawn_point)
                    # save to cache
                    _walker_spawn_points_cache.append((loc.x, loc.y, loc.z))

        # route planner mode
        self.route_mode = RouteMode.MANUAL_LAP
        if self.route_mode == RouteMode.MANUAL_LAP:
            self.routeplanner = ManualRoutePlanner(self.lap_spwan_point_wp, self.lap_spwan_point_wp, self.world,
                                                   resolution=2, plan=TOWN7_PLAN, use_section=True)

        # ego vehicle bp
        self.ego_bp = self._create_vehicle_bluepprint('vehicle.mini.cooper_s_2021')
        self.ego = None

        # Collision sensor
        self.collision_hist = []
         # collision history length
        self.collision_hist_l = 1
        self.collision_bp = self.bp_library.find('sensor.other.collision')
        self.collision_sensor = None

        # camera
        self.camera_img = None
        self.camera_trans = carla.Transform(carla.Location(x=1.18, z=1.7))
        self.camera_sensor_type = 'sensor.camera.rgb'
        if self.use_semantic_camera:
            self.camera_sensor_type = 'sensor.camera.semantic_segmentation'
        self.camera_bp = self.bp_library.find(self.camera_sensor_type)
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.camera_width))
        self.camera_bp.set_attribute('image_size_y', str(self.camera_height))
        if hasattr(self, 'camera_fov'):
            self.camera_bp.set_attribute('fov', str(self.camera_fov))
        self.camera_sensor = None

        self.record_video = kwargs.get('record_video', False)
        if self.record_video:
            self.obs_camera_bp = self.bp_library.find('sensor.camera.rgb')
            self.obs_camera_bp.set_attribute('image_size_x', '800')
            self.obs_camera_bp.set_attribute('image_size_y', '600')
            self.obs_frame_data_queue = Queue()

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # frame buffer
        self.img_buff = deque(maxlen=self.n_images)
        self.frame_data_queue = Queue()

        # action buffer
        self.num_past_actions = kwargs.get('n_past_actions', 10)
        self.actions_queue = deque(maxlen=self.num_past_actions)

        # control history
        self.store_history = self.record_video
        if self.store_history:
            self.throttle_hist = []
            self.brakes_hist = []
            self.steers_hist = []
            self.speed_hist = []
            self.lspeed_lon_hist = []
            self.original_dis = []

        self.spawn_batch = True
        # cache vehicle blueprints
        if self.spawn_batch:
            self.vehicle_bp_caches = {}
            for nw in self.number_of_wheels:
                self.vehicle_bp_caches[nw] = self._cache_vehicle_blueprints(number_of_wheels=nw)

        encoder_type = kwargs.get('encoder_type')
        if encoder_type is None:
            raise ValueError(f'unknown encoder_type {self.encoder_type}')

        grayscale = kwargs.get('grayscale', False)
        if encoder_type == 'CNN':
            if observation_size == camera_size:
                if grayscale:
                    self._transform_observation = self._transform_CNN_grayscale_observation_no_resize
                else:
                    self._transform_observation = self._transform_CNN_observation_no_resize
            else:
                if grayscale:
                    self._transform_observation = self._transform_CNN_grayscale_observation
                else:
                    self._transform_observation = self._transform_CNN_observation

            if self.n_images > 1:
                if grayscale:
                    # grayscale image, stack in new axis in place of channel
                    self._combine_observations = lambda obs_array: np.array(obs_array, dtype=self.obs_dtype)
                else:
                    # RGB image, stack in channel dimension
                    self._combine_observations = lambda obs_array: np.concatenate(obs_array, axis=-1, dtype=np.uint8)
            else:
                self._combine_observations = lambda obs_array: obs_array
        elif encoder_type == 'VAE':
            # VAE case
            self._transform_observation = self._transform_VAE_observation
            if self.n_images > 1:
                self._combine_observations = lambda obs_array: np.array(obs_array, dtype=np.float16)
            else:
                self._combine_observations = lambda obs_array: obs_array

        self.mean = np.array([0.4652, 0.4417, 0.3799])
        self.std = np.array([0.0946, 0.1767, 0.1865])

        self.z_steps = {}

    def reset(self):
        # Clear history if exist
        if self.store_history:
            self.throttle_hist.clear()
            self.brakes_hist.clear()
            self.steers_hist.clear()
            self.speed_hist.clear()
            self.lspeed_lon_hist.clear()
            self.original_dis.clear()

        self.current_action = None
        self.img_buff.clear()
        self.actions_queue.clear()

        self.current_lane_dis = 0

        # delete sensor, vehicles and walkers
        # self._clear_all_actors(['sensor.other.collision', self.camera_sensor_type, 'vehicle.*', 'controller.ai.walker', 'walker.*'])
        # self._clear_all_actors(['sensor.other.collision', self.camera_sensor_type, 'vehicle.*'])

        # Clear sensor objects
        if self.camera_sensor is not None:
            # not the first time
            self.camera_sensor.stop()
            self.collision_sensor.stop()

            destroy_commands = [
                carla.command.DestroyActor(self.ego.id),
                carla.command.DestroyActor(self.camera_sensor.id),
                carla.command.DestroyActor(self.collision_sensor.id)
            ]
            self.client.apply_batch_sync(destroy_commands, False)

        self.camera_sensor = None
        self.collision_sensor = None
        self.ego = None

        # clear image
        self.camera_img = None

        # Disable sync mode
        # self._set_synchronous_mode(False)

        # Get actors polygon list
        self.vehicle_polygons = []

        # Spawn the ego vehicle
        ego_spawn_times = 0
        spawn_transform_index, spawn_transform = self.routeplanner.get_random_spawn_point()
        if spawn_transform_index not in self.z_steps:
            self.z_steps[spawn_transform_index] = 0.1
        z_step = self.z_steps[spawn_transform_index]

        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                raise Exception(f'cannot spawn at {transform}. waypoint index is {self.routeplanner._checkpoint_waypoint_index}')

            transform = self._make_safe_spawn_transform(spawn_transform, z_step)

            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                z_step += 0.1
                self.z_steps[spawn_transform_index] = z_step
                time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
        self.collision_hist.clear()

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(self.frame_data_queue.put)

        if self.record_video:
            bound_x = 0.5 + self.ego.bounding_box.extent.x
            bound_y = 0.5 + self.ego.bounding_box.extent.y
            bound_z = 0.5 + self.ego.bounding_box.extent.z

            obs_camera_trans = carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0))
            self.obs_camera_sensor = self.world.spawn_actor(blueprint=self.obs_camera_bp, transform=obs_camera_trans,
                                                            attach_to=self.ego, attachment_type=carla.AttachmentType.SpringArm)
            self.obs_camera_sensor.listen(self.obs_frame_data_queue.put)

        # Update timesteps
        self.time_step = 1
        self.reset_step += 1

        self.frame = self.world.tick()

        # get route plan
        self.routeplanner.set_vehicle(self.ego)
        self.waypoints = self.routeplanner.run_step()

        for _ in range(self.num_past_actions):
            self.actions_queue.append(np.array([0, 0]))

        self.first_additional_state = np.ravel(np.array(self.actions_queue, dtype=np.float16))

        return self._get_obs()

    def step(self, action):
        acc = action[0]
        steer = action[1]

        if acc > 0:
            throttle = np.clip(acc, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc, 0, 1)

        self.ego.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))

        if self.store_history:
            self.throttle_hist.append(float(throttle))
            self.brakes_hist.append(float(brake))
            self.steers_hist.append(float(steer))

        self.current_action = (throttle, steer, brake)
        self.actions_queue.append(action)

        self.frame = self.world.tick()

        self.waypoints = self.routeplanner.run_step()
        self.current_waypoint = self.routeplanner.current_waypoint

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        info = {}
        info['additional_state'] = np.ravel(np.array(self.actions_queue, dtype=np.float16))

        return self._get_obs(), self._get_reward(), self._get_terminal(), info

    def render(self, mode='human'):
        if mode == 'human':
            if self.camera_img is None:
                raise Exception('self.camera_img is None')

            cv2.imshow('Carla environment', self.camera_img)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            if self.record_video:
                return self._retrieve_image_data(self.obs_frame_data_queue, use_semantic_mask=False)
            else:
                return self._get_observation_image()
        elif mode == 'observation':
            return self._transform_observation(self.camera_img)

    def _get_reward(self):
        """ Calculate the step reward. """
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        r_speed = -abs(speed - self.desired_speed)
        
        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer**2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        self.current_lane_dis, w = get_lane_dis_numba(self.waypoints, ego_x, ego_y)
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
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

        # cost for braking
        brake_cost = self.current_action[2]

        r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 1 - brake_cost*2

        if self.store_history:
            self.speed_hist.append(speed)
            self.lspeed_lon_hist.append(lspeed_lon)
            self.original_dis.append(self.current_lane_dis)

        return r

    def _get_reward_ppo(self):
        # lane distance reward
        ego_x, ego_y = get_pos(self.ego)
        self.current_lane_dis, w = get_lane_dis_numba(self.waypoints, ego_x, ego_y)
        non_nan_lane_dis = np.nan_to_num(self.current_lane_dis, posinf=self.out_lane_thres + 1, neginf=-(self.out_lane_thres + 1))
        d_norm = abs(non_nan_lane_dis) / self.out_lane_thres
        lane_centering_reward = 1 - d_norm

        # termination reward
        if abs(non_nan_lane_dis) > self.out_lane_thres or len(self.collision_hist) > 0:
            return -10

        # speed reward
        v = self.ego.get_velocity()
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        min_desired_speed = 0.8 * self.desired_speed
        max_desired_speed = 1.2 * self.desired_speed
        if lspeed_lon < min_desired_speed:
            speed_reward = lspeed_lon / min_desired_speed
        elif lspeed_lon > self.desired_speed:
            speed_reward = 1 - (lspeed_lon - self.desired_speed) / (max_desired_speed - self.desired_speed)
        else:
            speed_reward = 1

        # angle
        angle_degree = get_vehicle_angle(self.ego.get_transform(), self.current_waypoint.transform)
        # allow only 4 degree until no reward
        angle_reward = max(1.0 - abs(angle_degree / 4), 0.0)

        return speed_reward + lane_centering_reward + angle_reward

    def _get_terminal(self):
        """ Calculate whether to terminate the current episode. """
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            return True

        # If at destination
        if self.dests is not None: # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x-dest[0])**2 + (ego_y-dest[1])**2) < 4:
                    return True

        # If out of lane
        if abs(self.current_lane_dis) > self.out_lane_thres:
            return True

        # end of section
        if self.routeplanner.is_end_of_section:
            return True

        return False

    def _get_obs(self):
        self.camera_img = self._retrieve_image_data(self.frame_data_queue,
                                                    use_semantic_mask=self.use_semantic_camera)

        if self.n_images == 1:
            transformed_observation = self._transform_observation(self.camera_img)
            return self._combine_observations(transformed_observation)

        self.img_buff.append(self.camera_img)
        while len(self.img_buff) < self.n_images:
            self.img_buff.append(self.camera_img)

        img_array = [self._transform_observation(img) for img in self.img_buff]
        return self._combine_observations(img_array)

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
        actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
        bp: the blueprint object of carla.
        """
        blueprints = self.bp_library.filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _cache_vehicle_blueprints(self, number_of_wheels=4):
        if not isinstance(number_of_wheels, int):
            raise TypeError(f'number_of_wheels must be int not {type(number_of_wheels)}')

        blueprint_library = []
        blueprints = self.bp_library.filter('vehicle.*')
        for bp in blueprints:
            if bp.get_attribute('number_of_wheels').as_int() == number_of_wheels:
                if bp.has_attribute('color'):
                    color = random.choice(bp.get_attribute('color').recommended_values)
                    bp.set_attribute('color', color)

                blueprint_library.append(bp)

        return blueprint_library

    def _clear_all_actors(self, actor_filters):
        """ Clear specific actors. """
        destroy_commands = []
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                destroy_commands.append(carla.command.DestroyActor(actor))

        self.client.apply_batch(destroy_commands)

    def _set_synchronous_mode(self, synchronous = True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4], set_autopilot=True):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            if set_autopilot:
                vehicle.set_autopilot()
            return True, vehicle
        return False, vehicle

    def _spawn_random_vehicles_batch(self, transforms, number_of_vehicles, number_of_wheels=[4]):
        bps = []
        for nw in number_of_wheels:
            bps.extend(self.vehicle_bp_caches[nw])

        count = 0
        spawn_commands = []
        for transform in transforms:
            bp = random.choice(bps)
            bp.set_attribute('role_name', 'autopilot')
            spawn_cmd = carla.command.SpawnActor(bp, transform)
            spawn_cmd.then(carla.command.SetAutopilot(carla.command.FutureActor, True))
            spawn_commands.append(spawn_cmd)
            
            count += 1
            if count == number_of_vehicles:
                break

        self.client.apply_batch(spawn_commands)

        if count < number_of_vehicles:
            spawn_commands.clear()

            while count < number_of_vehicles:
                transform = random.choice(transforms)
                bp = random.choice(bps)
                bp.set_attribute('role_name', 'autopilot')
                spawn_cmd = carla.command.SpawnActor(bp, transform)
                spawn_cmd.then(carla.command.SetAutopilot(carla.command.FutureActor, True))
                spawn_commands.append(spawn_cmd)

                count += 1

            self.client.apply_batch(spawn_commands)

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.bp_library.filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.bp_library.find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _spwan_random_walkers_batch(self, transforms, number_of_walkers):
        walker_bps = self.bp_library.filter('walker.*')

        # spawn walker
        count = 0
        spawn_commands = []
        for transform in transforms:
            walker_bp = random.choice(walker_bps)

            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            spawn_commands.append(carla.command.SpawnActor(walker_bp, transform))

            count += 1
            if count == number_of_walkers:
                break

        results = self.client.apply_batch_sync(spawn_commands, True)
        walkers_list = []
        for result in results:
            if not result.error:
                walkers_list.append({'id': result.actor_id})

        # spawn controller
        spawn_commands.clear()
        walker_controller_bp = self.bp_library.find('controller.ai.walker')
        for i in range(len(walkers_list)):
            spawn_commands.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]['id']))

        results = self.client.apply_batch_sync(spawn_commands, True)
        controller_ids = []
        for i in range(len(results)):
            if not results[i].error:
                walkers_list[i]['con_id'] = results[i].actor_id
                controller_ids.append(results[i].actor_id)
        
        controller_actors = self.world.get_actors(controller_ids)

        self.world.wait_for_tick()

        # start controller
        con_idx = 0
        for walker in walkers_list:
            if 'con_id' not in walker:
                continue

            controller = controller_actors[con_idx]
            assert walker['con_id'] == controller.id

            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

            con_idx += 1

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
        filt: the filter indicating what type of actors we'll look at.

        Returns:
        actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l,w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R,poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
        transform: the carla transform object.
        Returns:
        Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        if self.vehicle_polygons:
            for idx, poly in self.vehicle_polygons[-1].items():
                poly_center = np.mean(poly, axis=0)
                ego_center = np.array([transform.location.x, transform.location.y])
                dis = np.linalg.norm(poly_center - ego_center)
                if dis > 8:
                    continue
                else:
                    overlap = True
                    break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _make_safe_spawn_transform(self, spawn_point_transform, z_step):
        ''' Set Z axis to 0.39 if Z axis of transform equals to 0.00 to prevent collision when spawning '''
        old_z = spawn_point_transform.location.z
        new_location = carla.Location(x=spawn_point_transform.location.x,
                                        y=spawn_point_transform.location.y,
                                        z=old_z + z_step)
        new_transform = carla.Transform(location=new_location, rotation=spawn_point_transform.rotation)

        return new_transform

    def _transform_CNN_observation(self, obs):
        cropped_obs = self._crop_image(obs)
        resized_obs = cv2.resize(cropped_obs, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)
        # normalized_obs = normalize_image(resized_obs, self.mean, self.std).astype(np.float16)

        return resized_obs

    def _transform_CNN_grayscale_observation(self, obs):
        cropped_obs = self._crop_image(obs)
        resized_obs = cv2.resize(cropped_obs, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)
        gray_obs = (cv2.cvtColor(resized_obs, cv2.COLOR_RGB2GRAY) / 255.).astype(np.float16)

        return gray_obs

    def _transform_CNN_observation_no_resize(self, obs):
        scaled_obs = obs / 255.
        return scaled_obs.transpose((2, 0, 1))

    def _transform_CNN_grayscale_observation_no_resize(self, obs):
        scaled_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) / 255.
        return scaled_obs

    def _transform_VAE_observation(self, obs):
        cropped_obs = self._crop_image(obs)
        resized_obs = cv2.resize(cropped_obs, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)
        normalized_obs = normalize_image(resized_obs, self.mean, self.std).astype(np.float16)

        return normalized_obs

    def _transform_old_VAE_observation(self, obs):
        ''' For old version that doesn't need normalization '''
        resized_obs = cv2.resize(obs, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)

        return (resized_obs / 255.0).astype(np.float16)

    # def _transform_observation(self, obs):
    #     return (obs / 255.0).astype(np.float16)

    def _get_observation_image(self):
        ''' Return RGB image in `H` x `W` x `C` format, its size match observation size.
        
            The difference between this method and `_transform_observation` is that `_transform_observation`
            will normalize an image but this method keeps the image format except spatial size.
        '''
        cropped_img = self._crop_image(self.camera_img)

        return cv2.resize(cropped_img, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)

    # def _get_observation_image(self):
    #     return self.camera_img

    def _crop_image(self, img):
        # this size is suitable for 800x600, fov 110 camera
        cropped_size = (384, 768)
        return center_crop(img, cropped_size, shift_H=1.2)

    def _retrieve_image_data(self, queue_to_wait, use_semantic_mask=False):
        while True:
            data = queue_to_wait.get()
            queue_to_wait.task_done()
            if data.frame == self.frame:
                break

        if use_semantic_mask:
            data.convert(cc.CityScapesPalette)
        else:
            data.convert(cc.Raw)
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]

        # BGR(OpenCV) > RGB
        return np.ascontiguousarray(array[:, :, ::-1])

    def _draw_debug_waypoints(self, waypoints, size=1, color=(255,0,0)):
        ''' Draw debug point on waypoints '''
        if len(waypoints) < 1:
            raise ValueError('number_of_waypoint must be greater than or equal to 1.')

        debug = self.world.debug
        color = carla.Color(r=color[0], g=color[1], b=color[2])
        for wp in waypoints:
            location = carla.Location(x=wp[0], y=wp[1], z=1.0)
            debug.draw_point(location, size=size, color=color)

    def close(self):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        if not self.dry_run:
            self._set_synchronous_mode(False)
            # delete all sensor for the next world
            self._clear_all_actors(['sensor.other.collision', self.camera_sensor_type, 'vehicle.*', 'controller.ai.walker', 'walker.*'])

        return super().close()

    def plot_control_graph(self, name):
        if not self.store_history:
            print('Cannot plot graph because environment does not store history')
            return

        data_np = np.array([self.throttle_hist, self.brakes_hist, self.steers_hist]).transpose()
        data = pd.DataFrame(data_np, columns=['throttle', 'brake', 'steer']).reset_index()
        data = pd.melt(data, id_vars='index', var_name='command', value_name='value')

        sns.lineplot(data=data, hue='command', x='index', y='value')
        plt.title('Throttle, Brake and Steer')
        plt.savefig(name)

    def plot_speed_graph(self, name):
        if not self.store_history:
            print('Cannot plot graph because environment does not store history')
            return

        data_np = np.array([self.speed_hist, self.lspeed_lon_hist]).transpose()
        data = pd.DataFrame(data_np, columns=['speed', 'speed_lon']).reset_index()
        data = pd.melt(data, id_vars='index', var_name='command', value_name='value')

        sns.lineplot(data=data, hue='command', x='index', y='value')
        plt.title('Speed and Longitudinal speed')
        plt.savefig(name)

    def plot_distance_graph(self, name):
        if not self.store_history:
            print('Cannot plot graph because environment does not store history')
            return

        data_np = np.array([self.original_dis]).transpose()
        data = pd.DataFrame(data_np, columns=['original distance']).reset_index()
        data = pd.melt(data, id_vars='index', var_name='distance', value_name='value')

        sns.lineplot(data=data, hue='distance', x='index', y='value')
        plt.title('Distance from center of the lane')
        plt.savefig(name)

    def get_latest_milestone(self):
        ''' Return index of latest checkpoint waypoint that the agent can go '''
        if self.routeplanner._intermediate_checkpoint_waypoint_index > self.routeplanner._checkpoint_waypoint_index:
            return self.routeplanner._checkpoint_waypoint_index
        else:
            return self.routeplanner._intermediate_checkpoint_waypoint_index

    @property
    def metadata(self):
        return {"render_modes": ["human", "rgb_array"], "render_fps": self.frame_per_second}

    def collect_env_images(self, num_steps, start_step=0, agent_class=BehaviorAgent, observation_callback=None):
        agent = agent_class(self.ego)

        for _ in range(start_step, start_step + num_steps):
            self.ego.apply_control(agent.run_step())

            self.world.tick()

            if observation_callback is not None:
                observation_callback(self._get_observation_image())

    def test_carla_agent(self, num_steps, start_step=0, recorder=None):
        agent = BasicAgent(self.ego)
        # agent = ManualAgent(self.ego)

        agent.set_global_plan(self.route_waypoints)
        agent.ignore_traffic_lights(active=True)
        agent.ignore_stop_signs(active=True)
        # agent.set_target_speed(40)

        for step in trange(start_step, start_step + num_steps):
            self.ego.apply_control(agent.run_step())

            self.world.tick()

            if recorder is not None:
                recorder.capture_frame()

            # img_np = self._get_observation_image()
            # img = Image.fromarray(img_np)
            # img.save(f'carla_town7_images/outskirts/town7_outskirts_{step:05d}.jpeg')
            # img.close()

            if agent.done():
                print('agent is done')
                break

        CarlaEnv.start_wp_idx += 5
        completed_lap = CarlaEnv.start_wp_idx > len(self.route_waypoints)

        return completed_lap


class SteerDirection(Enum):
    RIGHT = auto()
    LEFT = auto()


class ManualAgent:
    steer_right_cmds = [0.0] * 30 + [0.2] * 10 + [-0.1] * 15
    steer_left_cmds = [0.0] * 30 + [-0.05] * 5 + [-0.1] * 5

    def __init__(self, vehicle):
        steer_direction = SteerDirection.LEFT

        if steer_direction == SteerDirection.RIGHT:
            self.steer_cmds = deepcopy(ManualAgent.steer_right_cmds)
        elif steer_direction == SteerDirection.LEFT:
            self.steer_cmds = deepcopy(ManualAgent.steer_left_cmds)
        else:
            raise TypeError('unknown steer_direction type')

    def run_step(self):
        throttle = max(0.4, random.random())

        steer = 0
        if len(self.steer_cmds) > 0:
            steer = self.steer_cmds[0]
            self.steer_cmds.pop(0)

        return carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=0.0)

    def done(self):
        return False


class CarlaPerfectActionSampler:
    def __init__(self, env: CarlaEnv) -> None:
        self.agent = BasicAgent(env.ego,
                                target_speed=env.desired_speed * 3.6)

        self.agent.set_global_plan(env.routeplanner.get_route_waypoints())
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
