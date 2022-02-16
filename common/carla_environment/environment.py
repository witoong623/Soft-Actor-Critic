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
from enum import Enum, auto
from gym import spaces
from PIL import Image
from tqdm import trange

from .misc import set_carla_transform, get_pos, get_lane_dis
from .route_planner import RoutePlanner
from .manual_route_planner import ManualRoutePlanner, TOWN4_PLAN, TOWN4_REVERSE_PLAN

from agents.navigation.behavior_agent import BehaviorAgent


class RouteMode(Enum):
    BASIC_RANDOM = auto()
    MANUAL_LAP = auto()

_walker_spawn_points_cache = []
_load_world = False


class CarlaEnv(gym.Env):
    def __init__(self, **kwargs):
        global _load_world
        self.host = 'witoon-carla'
        self.port = 2000

        self.n_images = 2
        self.obs_width = 480
        self.obs_height = 270

        self.map = 'Town02'
        self.dt = 0.1
        self.frame_per_second = round(1 / self.dt)
        self.reload_world = True
        self.use_semantic_camera = True

        self.camera_width = 800
        self.camera_height = 600
        self.camera_fov = 110
        
        self.number_of_walkers = 50
        self.number_of_vehicles = 20
        self.number_of_wheels = [4]
        self.max_ego_spawn_times = 200
        self.max_time_episode = 1000
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

        self.dry_run = kwargs.get('dry_run', False)
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

        self.bp_library = self.world.get_blueprint_library()

        # spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # top left spawn point of town4
        self.lap_spwan_point_wp = self.world.get_map().get_waypoint(self.vehicle_spawn_points[1].location)
        # for collect images
        # self.lap_opposite_spwan_point_wp = self.lap_spwan_point_wp.get_left_lane()

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
            self.routeplanner = ManualRoutePlanner(self.lap_spwan_point_wp, self.lap_spwan_point_wp, resolution=5, plan=TOWN4_PLAN)
            # for collect images
            # self.routeplanner = ManualRoutePlanner(self.lap_opposite_spwan_point_wp, self.lap_opposite_spwan_point_wp, resolution=1, plan=TOWN4_REVERSE_PLAN)

        # ego vehicle bp
        self.ego_bp = self._create_vehicle_bluepprint('vehicle.nissan.micra', color='49,8,8')

        # Collision sensor
        self.collision_hist = []
         # collision history length
        self.collision_hist_l = 1
        self.collision_bp = self.bp_library.find('sensor.other.collision')

        # camera
        self.camera_img = None
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_sensor_type = 'sensor.camera.rgb'
        if self.use_semantic_camera:
            self.camera_sensor_type = 'sensor.camera.semantic_segmentation'
        self.camera_bp = self.bp_library.find(self.camera_sensor_type)
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.camera_width))
        self.camera_bp.set_attribute('image_size_y', str(self.camera_height))
        self.camera_bp.set_attribute('fov', str(self.camera_fov))

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # frame buffer
        self.img_buff = deque(maxlen=self.n_images)

        # action buffer
        self.num_past_actions = 10
        self.actions_queue = deque(maxlen=self.num_past_actions)

        # control history
        self.store_history = False
        if self.store_history:
            self.throttle_hist = []
            self.brakes_hist = []
            self.steers_hist = []
            self.speed_hist = []
            self.lspeed_lon_hist = []

        self.spawn_batch = True
        # cache vehicle blueprints
        if self.spawn_batch:
            self.vehicle_bp_caches = {}
            for nw in self.number_of_wheels:
                self.vehicle_bp_caches[nw] = self._cache_vehicle_blueprints(number_of_wheels=nw)

        # termination condition
        self.terminate_on_out_of_lane = True

    def reset(self):
        # Clear history if exist
        if self.store_history:
            self.throttle_hist.clear()
            self.brakes_hist.clear()
            self.steers_hist.clear()

        # Clear sensor objects
        self.camera_sensor = None
        self.collision_sensor = None

        # delete sensor, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', self.camera_sensor_type, 'vehicle.*', 'controller.ai.walker', 'walker.*'])

        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        if self.spawn_batch and self.number_of_vehicles > 0:
            self._spawn_random_vehicles_batch(self.vehicle_spawn_points, self.number_of_vehicles, number_of_wheels=self.number_of_wheels)
        else:
            count = self.number_of_vehicles
            vehicles = []
            if count > 0:
                for spawn_point in self.vehicle_spawn_points:
                    success_spwan, vehicle = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=self.number_of_wheels)
                    if success_spwan:
                        count -= 1
                        vehicles.append(vehicle)
                    if count <= 0:
                        break
            while count > 0:
                success_spwan, vehicle = self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=self.number_of_wheels)
                if success_spwan:
                    count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        if self.spawn_batch and self.number_of_walkers > 0:
            self._spwan_random_walkers_batch(self.walker_spawn_points, self.number_of_walkers)
        else:
            count = self.number_of_walkers
            if count > 0:
                for spawn_point in self.walker_spawn_points:
                    if self._try_spawn_random_walker_at(spawn_point):
                        count -= 1
                    if count <= 0:
                        break
            while count > 0:
                if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                    count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()

            if self.route_mode == RouteMode.BASIC_RANDOM:
                if self.task_mode == 'random':
                    transform = random.choice(self.vehicle_spawn_points)
                if self.task_mode == 'roundabout':
                    self.start = [52.1 + np.random.uniform(-5,5), -4.2, 178.66] # random
                    # self.start=[52.1,-4.2, 178.66] # static
                    transform = set_carla_transform(self.start)
            elif self.route_mode == RouteMode.MANUAL_LAP:
                transform = self.routeplanner.spawn_transform

            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
        if len(self.collision_hist) > self.collision_hist_l:
            self.collision_hist.pop(0)
        self.collision_hist = []

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))
        def get_camera_img(data):
            if self.use_semantic_camera:
                data.convert(cc.CityScapesPalette)
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            # array = cv2.resize(array, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)

            # BGR(OpenCV) > RGB
            array = np.ascontiguousarray(array[:, :, ::-1])
            self.camera_img = array

        # wait for camera image from sensor
        while self.camera_img is None:
            time.sleep(1)

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        if self.route_mode == RouteMode.BASIC_RANDOM:
            self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
            self.waypoints = self.routeplanner.run_step()
        elif self.route_mode == RouteMode.MANUAL_LAP:
            self.routeplanner.set_vehicle(self.ego)
            self.waypoints = self.routeplanner.run_step()

        for _ in range(self.num_past_actions):
            self.actions_queue.append(np.array([0, 0]))

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

        self.world.tick()

        self.waypoints = self.routeplanner.run_step()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        info = {}
        info['past_actions'] = np.array(self.actions_queue)
        self.actions_queue.append(action)
        info['next_past_actions'] = np.array(self.actions_queue)

        return self._get_obs(), self._get_reward(), self._get_terminal(), info

    def render(self, mode='human'):
        if mode == 'human':
            if self.camera_img is None:
                print('self.camera_img is None')
                raise Exception('self.camera_img is None')

            cv2.imshow('Carla environment', self.camera_img)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.camera_img

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
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -100

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
            r_fast = -1

        # if it is faster than desired speed, minus the excess speed
        # and don't give reward from speed
        r_fast *= lspeed_lon

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

        r = 200 * r_collision + 1 * lspeed_lon + r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1

        if self.store_history:
            self.speed_hist.append(speed)
            self.lspeed_lon_hist.append(lspeed_lon)

        return r

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

        # if complete a number of laps
        if self.route_mode == RouteMode.MANUAL_LAP:
            if self.routeplanner.lap_count >= 2:
                return True

        # If at destination
        if self.dests is not None: # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x-dest[0])**2 + (ego_y-dest[1])**2) < 4:
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres and self.terminate_on_out_of_lane:
            return True

        return False

    def _get_obs(self):
        if self.n_images == 1:
            return self._transform_observation(self.camera_img)

        self.img_buff.append(self.camera_img)
        while len(self.img_buff) < self.n_images:
            self.img_buff.append(self.camera_img)

        img_array = [self._transform_observation(img) for img in self.img_buff]
        # return np.array(img_array, dtype=np.float32)
        # TODO: for CNN, concatenate it
        return np.concatenate(img_array, axis=0)

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

    def _transform_observation(self, obs):
        ''' Transform image observation to specified observation size and normalize it '''
        obs = cv2.resize(obs, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)
        # TODO: for CNN encoder, make it to C x H x W
        obs = obs.transpose((2, 0, 1))
        return obs / 255.

    def _get_image(self):
        ''' Return RGB image in `H` x `W` x `C` format, its size match observation size. '''
        return cv2.resize(self.camera_img, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)

    def close(self):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        if not self.dry_run:
            # delete all sensor for the next world
            self._clear_all_actors(['sensor.other.collision', self.camera_sensor_type, 'vehicle.*', 'controller.ai.walker', 'walker.*'])

        return super().close()

    def plot_control_graph(self, name):
        if not self.store_history:
            raise Exception('Cannot plot graph because environment does not store history')

        data_np = np.array([self.throttle_hist, self.brakes_hist, self.steers_hist]).transpose()
        data = pd.DataFrame(data_np, columns=['throttle', 'brake', 'steer']).reset_index()
        data = pd.melt(data, id_vars='index', var_name='command', value_name='value')

        sns.lineplot(data=data, hue='command', x='index', y='value')
        plt.title('Throttle, Brake and Steer')
        plt.savefig(name)

    def plot_speed_graph(self, name):
        if not self.store_history:
            raise Exception('Cannot plot graph because environment does not store history')

        data_np = np.array([self.speed_hist, self.lspeed_lon_hist]).transpose()
        data = pd.DataFrame(data_np, columns=['speed', 'speed_lon']).reset_index()
        data = pd.melt(data, id_vars='index', var_name='command', value_name='value')

        sns.lineplot(data=data, hue='command', x='index', y='value')
        plt.title('Speed and Longitudinal speed')
        plt.savefig(name)

    @property
    def metadata(self):
        return {"render.modes": ["human", "rgb_array"], "video.frames_per_second": self.frame_per_second}

    def collect_env_images(self, num_steps, observation_callback=None):
        agent = BehaviorAgent(self.ego)

        for _ in range(num_steps):
            self.ego.apply_control(agent.run_step())

            self.world.tick()

            if observation_callback is not None:
                observation_callback(self._get_image())

    def test_carla_agent(self, num_steps, recorder):

        agent = BasicAgent(self.ego)

        route_waypoints = self.routeplanner.get_route_waypoints()
        agent.set_global_plan(route_waypoints)
        agent.ignore_traffic_lights(active=True)
        agent.ignore_stop_signs(active=True)

        start = 0
        for step in trange(start, num_steps + start):
            self.ego.apply_control(agent.run_step())

            self.world.tick()

            recorder.capture_frame()

            # img_np = self._get_image()
            # img = Image.fromarray(img_np)
            # img.save(f'carla_town7_images/outskirts/town7_outskirts_{step:04d}.jpeg')
            # img.close()

            if agent.done():
                print('agent is done')
                break
