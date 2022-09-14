# copy from https://github.com/bitsauce/Carla-ppo/blob/master/CarlaEnv/planner.py
import carla
import numpy as np

from numba.typed import List
from collections import deque
from agents.navigation.local_planner import RoadOption
from common.carla_environment.custom_route_planner import AITRoutePlanner, Town7RoutePlanner
from common.carla_environment.checkpoints_manager import Town7CheckpointManager, AITCheckpointManager
from common.carla_environment.misc import get_pos


# cache for entire lifecycle of application
_route_transform = None
_transformed_waypoint_routes = None


def carla_to_vector(obj):
    ''' Turn Carla object which have some kind of coordinate attributes to `np.ndarray` '''
    if isinstance(obj, carla.Location) or isinstance(obj, carla.Vector3D):
        return np.array([obj.x, obj.y, obj.z])
    elif isinstance(obj, carla.Rotation):
        return np.array([obj.pitch, obj.yaw, obj.roll])
    else:
        raise TypeError(f'obj must be `Location`, `Vector3D` or `Rotation`, not {type(obj)}')


class RouteTracker:
    def __init__(self, start_waypoint, end_waypoint, world, resolution=2.0,
                 initial_checkpoint=0, repeat_section_threshold=5,
                 use_section=False, enable=True, debug_route_waypoint_len=None,
                 traffic_mode='RHT'):
        ''' `route_waypoint_len` is purely for testing purpose '''
        global _route_transform, _transformed_waypoint_routes

        self._vehicle = None
        self._world = world
        self._map = world.get_map()

        self.start_waypoint = start_waypoint
        self.end_waypoint = end_waypoint

        self._check_passed_waypoint = self._check_passed_waypoint_RHT if traffic_mode == 'RHT' else self._check_passed_waypoint_LHT

        self._current_waypoint_index = initial_checkpoint

        if enable:
            self.carla_debug = self._world.debug

            if self._is_AIT_map():
                self.route_planner = AITRoutePlanner(self._world, resolution)
            else:
                self.route_planner = Town7RoutePlanner(self._world, resolution,
                                                       self.start_waypoint, self.end_waypoint, TOWN7_PLAN)
            
            _route_transform = self.route_planner.compute_route()
            _transformed_waypoint_routes = List(self._transform_transforms(_route_transform))

        route_waypoint_len = len(_route_transform) if debug_route_waypoint_len is None else debug_route_waypoint_len

        if self._is_AIT_map():
            self.checkpoint_manager = AITCheckpointManager(route_waypoint_len,
                                                           repeat_section_threshold,
                                                           initial_checkpoint)
        else:
            self.checkpoint_manager = Town7CheckpointManager(route_waypoint_len,
                                                             repeat_section_threshold,
                                                             initial_checkpoint)

        self.in_junction = False
        self.angle_entering_junction = 0
        self.turning_window_len = 10
        self.turning_window = deque(maxlen=self.turning_window_len)
        self.is_correct_direction_cache = {}

    def set_vehicle(self, vehicle):
        ''' Set internal state to current vehicle, must be called in `reset` '''
        self._vehicle = vehicle
        self.is_correct_direction_cache.clear()

    def run_step(self):
        waypoint_routes_len = len(_route_transform)
        current_transform = self._vehicle.get_transform()
        waypoint_index = self._current_waypoint_index
        for _ in range(waypoint_routes_len):
            # check if we passed next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            trans, _ = _route_transform[next_waypoint_index % waypoint_routes_len]
            dot_ret = np.dot(carla_to_vector(trans.get_forward_vector())[:2],
                             carla_to_vector(current_transform.location - trans.location)[:2])

            if self._check_passed_waypoint(dot_ret):
                waypoint_index += 1
            else:
                break

        self._current_waypoint_index = waypoint_index % waypoint_routes_len

        self.checkpoint_manager.update_checkpoint(self._current_waypoint_index)

        self.completed_lap = self.checkpoint_manager.does_complete_lap(self._current_waypoint_index)

        return _transformed_waypoint_routes[self._current_waypoint_index:]

    def get_route_waypoints(self):
        ''' Return list of (waypoint, RoadOption) '''
        raise Exception('route waypoints are not available')

    def get_transformed_route_waypoints(self):
        return _transformed_waypoint_routes

    def get_spawn_point(self, specified_index=None):
        if specified_index is not None:
            index = specified_index
        else:
            index = self.checkpoint_manager.get_spawn_point_index()

        self._current_waypoint_index = index

        if self._is_AIT_map():
            transform = self.checkpoint_manager.get_correct_spawn_point_transform(_route_transform[index][0], index)
        else:
            transform = _route_transform[index][0]

        return index, transform

    def get_angle_to_next_section_waypoint(self):
        x, y = get_pos(self._vehicle)
        next_transform = self.route_planner.get_first_next_section_transform(self._current_waypoint_index)
        next_x, next_y = next_transform.location.x, next_transform.location.y
        corr_dirrection_vec = np.array([next_x - x, next_y - y])
        corr_norm = np.linalg.norm(corr_dirrection_vec)
        corr_dirrection_vec /= corr_norm

        carla_forward_vector = self._vehicle.get_transform().get_forward_vector().make_unit_vector()
        forward_vector = np.array([carla_forward_vector.x, carla_forward_vector.y])
        angle_radian = np.arctan2(np.cross(corr_dirrection_vec, forward_vector), np.dot(corr_dirrection_vec, forward_vector))

        if angle_radian > np.pi:
            angle_radian -= 2 * np.pi
        elif angle_radian <= -np.pi:
            angle_radian += 2 * np.pi

        return np.rad2deg(angle_radian)

    LEFT_ANGLE_LIMIT = -25
    RIGHT_ANGLE_LIMIT = 23

    def is_correct_direction(self, frame_number):
        if frame_number in self.is_correct_direction_cache:
            return self.is_correct_direction_cache[frame_number]

        is_correct_direction = self._is_correct_direction()

        self.is_correct_direction_cache[frame_number] = is_correct_direction

        return is_correct_direction

    def is_in_junction(self):
        is_in_junction = self.route_planner.is_junction_waypoint(self._current_waypoint_index)

        if is_in_junction:
            if not self.in_junction:
                self.in_junction = True
                self.angle_entering_junction = self.get_angle_to_next_section_waypoint()
        else:
            if self.in_junction:
                self.in_junction = False
                self.angle_entering_junction = 0
                self.turning_window.clear()

        return is_in_junction

    def get_bonus_out_of_lane_distance(self):
        return self.route_planner.get_bonus_out_of_lane_distance(self._current_waypoint_index)

    @property
    def is_end_of_section(self):
        return self.checkpoint_manager.is_end_of_section(self._current_waypoint_index)

    def _is_correct_direction(self):
        # only work within junction, so always return True if it is outside junction
        is_in_junction = self.route_planner.is_junction_waypoint(self._current_waypoint_index)
        if not is_in_junction:
            return True

        angle_degree = self.get_angle_to_next_section_waypoint()

        self.turning_window.append(angle_degree)

        if len(self.turning_window) < self.turning_window.maxlen:
            return True
        else:
            avg_angle = sum(self.turning_window) / self.turning_window.maxlen

            if self.angle_entering_junction < 0:
                angle_less_than_enter = avg_angle > self.angle_entering_junction
            else:
                angle_less_than_enter = avg_angle < self.angle_entering_junction

            if angle_less_than_enter:
                return True

            return self.LEFT_ANGLE_LIMIT <= angle_degree <= self.RIGHT_ANGLE_LIMIT

    def _transform_transforms(self, transforms):
        ''' Transform a waypoint into list of x, y and yaw '''
        return list(map(lambda tf: (tf[0].location.x, tf[0].location.y, tf[0].rotation.yaw), transforms))

    def _get_AIT_spawn_point(self):
        return 0, _route_transform[0][0]

    def _is_AIT_map(self):
        return self._map.name == 'ait_v4/Maps/ait_v4/ait_v4'

    def _check_passed_waypoint_RHT(self, dot_product_result):
        return dot_product_result > 0.0

    def _check_passed_waypoint_LHT(self, dot_product_result):
        return dot_product_result < 0.0

    def _draw_debug_waypoint(self, waypoint):
        self.carla_debug.draw_point(waypoint.transform.location, size=0.3, life_time=60)

    def _draw_forward_vector(self, transform):
        forward_vector = transform.get_forward_vector()
        begin = transform.location
        end = transform.location + forward_vector

        self.carla_debug.draw_arrow(begin, end, thickness=0.1, arrow_size=0.2, life_time=120)


TOWN7_PLAN = [RoadOption.STRAIGHT] + [RoadOption.RIGHT] * 2 + [RoadOption.STRAIGHT] * 5
