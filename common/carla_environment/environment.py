import carla
import cv2
import gym
import random
import time

import numpy as np

from carla import ColorConverter as cc
from collections import deque
from gym import spaces
from PIL import Image

from .misc import set_carla_transform, get_pos, get_lane_dis
from .route_planner import RoutePlanner


class CarlaEnv(gym.Env):
    def __init__(self, **params):
        self.host = 'witoon-carla'
        self.port = 2000

        self.n_images = 1
        self.obs_width = 480
        self.obs_height = 270

        self.map = 'Town03'
        self.dt = 0.1
        self.reload_world = True
        self.use_semantic_camera = True

        self.camera_width = 800
        self.camera_height = 600
        self.camera_fov = 110
        
        self.number_of_walkers = 0
        self.number_of_vehicles = 10
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

        print('connecting to Carla server...')
        client = carla.Client(self.host, self.port)
        client.set_timeout(20.0)
        if self.reload_world:
            self.world = client.load_world(self.map)
        else:
            self.world = client.get_world()
        print('Carla server connected!')

        # spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # ego vehicle bp
        self.ego_bp = self._create_vehicle_bluepprint('vehicle.nissan.micra', color='49,8,8')

        # Collision sensor
        self.collision_hist = []
         # collision history length
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # camera
        # self.camera_img = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        if self.use_semantic_camera:
            self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        else:
            self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
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
        self.camera_img = None
        self.img_buff = deque(maxlen=self.n_images)

    def reset(self):
        self.camera_sensor = None

        # delete sensor, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.camera.semantic_segmentation', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
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

            if self.task_mode == 'random':
                transform = random.choice(self.vehicle_spawn_points)
            if self.task_mode == 'roundabout':
                self.start = [52.1 + np.random.uniform(-5,5), -4.2, 178.66] # random
                # self.start=[52.1,-4.2, 178.66] # static
                transform = set_carla_transform(self.start)
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
            array = cv2.resize(array, (self.obs_width, self.obs_height), interpolation=cv2.INTER_NEAREST)

            # BGR(OpenCV) > RGB
            array = array[:, :, ::-1]
            self.camera_img = array

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.world.tick()

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints = self.routeplanner.run_step()

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

        self.world.tick()

        self.waypoints = self.routeplanner.run_step()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return self._get_obs(), self._get_reward(), self._get_terminal(), {}

    def render(self, mode='human'):
        if mode == 'human':
            if self.camera_img is None:
                print('self.camera_img is None')
                raise Exception('self.camera_img is None')

            cv2.imshow('Carla environment', self.camera_img)
            cv2.waitKey(1)
        elif mode == 'rgb':
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
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

        r = 200 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1

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

        # If at destination
        if self.dests is not None: # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x-dest[0])**2 + (ego_y-dest[1])**2) < 4:
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            return True

        return False

    def _get_obs(self):
        if self.n_images == 1:
            return self.camera_img

        self.img_buff.append(self.camera_img)
        while len(self.img_buff) < self.n_images:
            self.img_buff.append(self.camera_img)

        # print([type(o) for o in self.img_buff])

        return np.array(self.img_buff)

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
        actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
        bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _clear_all_actors(self, actor_filters):
        """ Clear specific actors. """
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                actor.destroy()

    def _set_synchronous_mode(self, synchronous = True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
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
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

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

    def close(self):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        return super().close()


# register(
#     id='carla-v0',
#     entry_point='carla_environment:CarlaEnv',
# )


if __name__ == '__main__':
    # env = gym.make('carla-v0')
    env = CarlaEnv()
    # print('action_space', env.action_space)
    # print('action_space.shape', env.action_space.shape)
    # print('observation_space', env.observation_space)
    obs = env.reset()

    excep_count = 1
    for i in range(50):
        new_obs, reward, done, info = env.step([0.5, 0])
        img = Image.fromarray(env.camera_img)
        img.save(f'carla_images/step_{i+1}.jpeg')
        # env.render()
        obs = new_obs

        if done:
            print('done')
            break

    key = input('pass anykey to exit')
    env.close()
