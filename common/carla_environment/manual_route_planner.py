# copy from https://github.com/bitsauce/Carla-ppo/blob/master/CarlaEnv/planner.py
import carla
import functools
import operator
import random
import numpy as np

from enum import Enum
from numba.typed import List
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import vector
from common.carla_environment.ait_route_planner import AITRoutePlanner
from common.carla_environment.checkpoints_manager import Town7CheckpointManager, AITCheckpointManager


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


class ManualRoutePlanner:
    def __init__(self, start_waypoint, end_waypoint, world, resolution=2.0,
                 plan=None, initial_checkpoint=0, repeat_section_threshold=5,
                 use_section=False, enable=True, debug_route_waypoint_len=None,
                 traffic_mode='RHT'):
        ''' `route_waypoint_len` is purely for testing purpose '''
        global _route_transform, _transformed_waypoint_routes

        self._vehicle = None
        self._world = world
        self._map = world.get_map()
        self.plan = plan
        self.traffic_mode = traffic_mode

        self._sampling_radius = resolution
        self._min_distance = self._sampling_radius - 1 if self._sampling_radius > 1 else 1

        self.start_waypoint = start_waypoint
        self.end_waypoint = end_waypoint

        self._check_pass_waypoint_func = lambda v: v > 0.0 if traffic_mode == 'RHT' else lambda v: v < 0.0

        self._current_waypoint_index = initial_checkpoint

        if enable:
            self.carla_debug = self._world.debug

            if self._is_AIT_map():
                ait_route_planner = AITRoutePlanner(self._world, resolution)
                _route_transform = ait_route_planner.compute_route_waypoints()
            else:
                _route_transform = self._compute_route_waypoints()
            
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

    def set_vehicle(self, vehicle):
        ''' Set internal state to current vehicle, must be called in `reset` '''
        self._vehicle = vehicle

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

            if self._check_pass_waypoint_func(dot_ret):
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

    def get_spawn_point(self):
        index = self.checkpoint_manager.get_spawn_point_index()
        self._current_waypoint_index = index

        transform = _route_transform[index][0]
        # if self._is_AIT_map():
        #     transform.rotation.yaw = 180

        return index, transform

    def _compute_route_waypoints(self):
        if self.plan is None:
            grp = GlobalRoutePlanner(self._map, self._sampling_radius)
            
            route = grp.trace_route(
                self.start_waypoint.transform.location,
                self.end_waypoint.transform.location)
        else:
            route = []
            current_waypoint = self.start_waypoint
            for i, action in enumerate(self.plan):
                # Generate waypoints to next junction
                wp_choice = [current_waypoint]
                while len(wp_choice) == 1:
                    current_waypoint = wp_choice[0]
                    route.append((current_waypoint, RoadOption.LANEFOLLOW))
                    wp_choice = current_waypoint.next(self._sampling_radius)

                    # Stop at destination
                    if i > 0 and current_waypoint.transform.location.distance(self.end_waypoint.transform.location) < self._sampling_radius:
                        break

                if action == RoadOption.VOID:
                    break

                # Make sure that next intersection waypoints are far enough
                # from each other so we choose the correct path
                step = self._sampling_radius
                while len(wp_choice) > 1:
                    wp_choice = current_waypoint.next(step)
                    wp0, wp1 = wp_choice[:2]
                    if wp0.transform.location.distance(wp1.transform.location) < self._sampling_radius:
                        step += self._sampling_radius
                    else:
                        break

                # Select appropriate path at the junction
                if len(wp_choice) > 1:
                    # Current heading vector
                    current_transform = current_waypoint.transform
                    current_location = current_transform.location
                    projected_location = current_location + \
                        carla.Location(
                            x=np.cos(np.radians(current_transform.rotation.yaw)),
                            y=np.sin(np.radians(current_transform.rotation.yaw)))
                    v_current = vector(current_location, projected_location)

                    direction = 0
                    if action == RoadOption.LEFT:
                        direction = 1
                    elif action == RoadOption.RIGHT:
                        direction = -1
                    elif action == RoadOption.STRAIGHT:
                        direction = 0
                    select_criteria = float("inf")

                    # Choose correct path
                    for wp_select in wp_choice:
                        v_select = vector(
                            current_location, wp_select.transform.location)
                        cross = float("inf")
                        if direction == 0:
                            cross = abs(np.cross(v_current, v_select)[-1])
                        else:
                            cross = direction * np.cross(v_current, v_select)[-1]
                        if cross < select_criteria:
                            select_criteria = cross
                            current_waypoint = wp_select

                    # Generate all waypoints within the junction
                    # along selected path
                    route.append((current_waypoint, action))
                    current_waypoint = current_waypoint.next(self._sampling_radius)[0]
                    while current_waypoint.is_intersection:
                        route.append((current_waypoint, action))
                        current_waypoint = current_waypoint.next(self._sampling_radius)[0]
            assert route

        # Change action 5 wp before intersection
        num_wp_to_extend_actions_with = 5
        action = route[0][1]
        for i in range(1, len(route)):
            next_action = route[i][1]
            if next_action != action:
                if next_action != RoadOption.LANEFOLLOW:
                    for j in range(num_wp_to_extend_actions_with):
                        route[i-j-1] = (route[i-j-1][0], route[i][1])
            action = next_action

        return route

    def _transform_transforms(self, transforms):
        ''' Transform a waypoint into list of x, y and yaw '''
        return list(map(lambda tf: (tf[0].location.x, tf[0].location.y, tf[0].rotation.yaw), transforms))

    def _get_AIT_spawn_point(self):
        return 0, _route_transform[0][0]

    def _is_AIT_map(self):
        return self._map.name == 'ait_v4/Maps/ait_v4/ait_v4'

    def _get_waypoint_forward_vector_np(self, waypoint):
        forward_vector = waypoint.transform.get_forward_vector()
        vector_np = np.array([forward_vector.x, forward_vector.y, forward_vector.z])

        if self.traffic_mode == 'LHT':
            vector_np = vector_np * -1

        return vector_np

    def _draw_debug_waypoint(self, waypoint):
        self.carla_debug.draw_point(waypoint.transform.location, size=0.3, life_time=60)

    @property
    def is_end_of_section(self):
        return self.checkpoint_manager.is_end_of_section(self._current_waypoint_index)


TOWN7_PLAN = [RoadOption.STRAIGHT] + [RoadOption.RIGHT] * 2 + [RoadOption.STRAIGHT] * 5
