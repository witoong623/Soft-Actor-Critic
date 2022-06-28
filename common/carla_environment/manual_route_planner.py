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

# cache waypoint for entire lifecycle of application
_route_waypoints = None
_transformed_waypoint_routes = None


def carla_to_vector(obj):
    ''' Turn Carla object which have some kind of coordinate attributes to `np.ndarray` '''
    if isinstance(obj, carla.Location) or isinstance(obj, carla.Vector3D):
        return np.array([obj.x, obj.y, obj.z])
    elif isinstance(obj, carla.Rotation):
        return np.array([obj.pitch, obj.yaw, obj.roll])
    else:
        raise TypeError(f'obj must be `Location`, `Vector3D` or `Rotation`, not {type(obj)}')


class NextFunction(Enum):
    NEXT = 1
    PREVIOUS = 2

    def opposite(self):
        if self.name == 'NEXT':
            return NextFunction.PREVIOUS
        else:
            return NextFunction.NEXT


class ManualRoutePlanner:
    def __init__(self, start_waypoint, end_waypoint, world, resolution=2.0,
                 plan=None, initial_checkpoint=0, repeat_section_threshold=5,
                 use_section=False, enable=True, debug_route_waypoint_len=None,
                 traffic_mode='RHT'):
        ''' `route_waypoint_len` is purely for testing purpose '''
        global _route_waypoints, _transformed_waypoint_routes

        self._vehicle = None
        self._world = world
        self._map = world.get_map()
        self.plan = plan
        self.traffic_mode = traffic_mode

        self._sampling_radius = resolution
        self._min_distance = self._sampling_radius - 1 if self._sampling_radius > 1 else 1

        self.start_waypoint = start_waypoint
        self.end_waypoint = end_waypoint

        self.lap_count = 0
        self._repeat_count = 0
        self._repeat_count_threshold = repeat_section_threshold
        self._checkpoint_frequency = 25
        self._check_pass_waypoint_func = lambda v: v > 0.0 if traffic_mode == 'RHT' else lambda v: v < 0.0

        self._checkpoint_waypoint_index = initial_checkpoint
        self._start_waypoint_index = self._checkpoint_waypoint_index
        self._current_waypoint_index = self._checkpoint_waypoint_index
        self._next_checkpoint_waypoint_index = self._checkpoint_waypoint_index + self._checkpoint_frequency

        if enable:
            self.carla_debug = self._world.debug

            if self._is_AIT_map():
                ait_route_planner = AITRoutePlanner(self._world, resolution)
                _route_waypoints = ait_route_planner.compute_route_waypoints()
            else:
                _route_waypoints = self._compute_route_waypoints()
            
            _transformed_waypoint_routes = List(self._transform_waypoints(_route_waypoints))

        self.completed_lap = False
        self._in_random_spawn_point = False
        self._reached_last_waypoint_index = False

        # for section checkpoint
        self.use_section = use_section
        if self.use_section:
            route_waypoint_len = len(_route_waypoints) if debug_route_waypoint_len is None else debug_route_waypoint_len
            # (start, end, checkpoint frequency)
            self.sections_indexes = [(0, 140, 35), (143, 173, 30), (176, route_waypoint_len - 1, 35)]
            self.sections_start = [s[0] for s in self.sections_indexes]
            self.sections_end = [s[1] for s in self.sections_indexes]
            self.sections_frequency = [s[2] for s in self.sections_indexes]
            self.sections_ends = [140, 141, 142, 173, 174, 175, 591]

            def get_all_indexes(start, end, step):
                indexes = []
                idx = start

                while not indexes or indexes[-1] < end:
                    idx = max(idx + step, end)
                    indexes.append(idx)

                return indexes

            self.all_spawn_indexes = functools.reduce(operator.concat,
                                                      [get_all_indexes(*sec) for sec in self.sections_indexes])
            self.round_spawn_idx = 0

            if initial_checkpoint < self.sections_end[0]:
                frequency = self.sections_indexes[0][2]
            elif initial_checkpoint < self.sections_end[1]:
                frequency = self.sections_indexes[1][2]
            elif initial_checkpoint < self.sections_end[2]:
                frequency = self.sections_indexes[2][2]

            self._next_checkpoint_waypoint_index = self._checkpoint_waypoint_index + frequency
            if self._next_checkpoint_waypoint_index > route_waypoint_len - 1:
                self._next_checkpoint_waypoint_index = 0

    def set_vehicle(self, vehicle):
        ''' Set internal state to current vehicle, must be called in `reset` '''
        self._vehicle = vehicle

    def run_step(self):
        waypoint_routes_len = len(_route_waypoints)
        current_transform = self._vehicle.get_transform()
        waypoint_index = self._current_waypoint_index
        for _ in range(waypoint_routes_len):
            # check if we passed next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = _route_waypoints[next_waypoint_index % waypoint_routes_len]
            dot_ret = np.dot(carla_to_vector(wp.transform.get_forward_vector())[:2],
                             carla_to_vector(current_transform.location - wp.transform.location)[:2])

            if self._check_pass_waypoint_func(dot_ret):
                waypoint_index += 1
            else:
                break

        self._current_waypoint_index = waypoint_index % waypoint_routes_len

        # update checkpoint
        # self._checkpoint_waypoint_index = (self._current_waypoint_index // self._checkpoint_frequency) * self._checkpoint_frequency
        if self.use_section and not self._in_random_spawn_point:
            self._update_checkpoint_by_section()

        if not self._reached_last_waypoint_index:
            self._reached_last_waypoint_index = self._update_reached_last_waypoint_index(waypoint_index)

        if self._reached_last_waypoint_index and not self.completed_lap:
            self.completed_lap = self._checkpoint_waypoint_index == 0

        return _transformed_waypoint_routes[self._current_waypoint_index:]

    def get_route_waypoints(self):
        ''' Return list of (waypoint, RoadOption) '''
        return _route_waypoints

    def get_transformed_route_waypoints(self):
        return _transformed_waypoint_routes

    def get_spawn_point(self):
        ''' get spawn point from pre-defined strategy '''
        if self._is_AIT_map():
            idx, transform = self._get_AIT_spawn_point()
        else:
            if not self.completed_lap:
                idx, transform = self._get_random_spawn_point()
            else:
                idx, transform = self._get_cycle_spawn_point()

        self._current_waypoint_index = idx
        self._start_waypoint_index = idx

        return idx, transform

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

    def _transform_waypoints(self, waypoints):
        ''' Transform a waypoint into list of x, y and yaw '''
        return list(map(lambda wp: (wp[0].transform.location.x, wp[0].transform.location.y, wp[0].transform.rotation.yaw), waypoints))

    def _update_checkpoint(self):
        ''' implement checkpoint logic that encourage the agent to remember more past road before trying next portion of the road '''
        idx = (self._current_waypoint_index // self._checkpoint_frequency) * self._checkpoint_frequency

        if idx >= self._next_checkpoint_waypoint_index:
            self._repeat_count += 1

            if self._repeat_count >= self._repeat_count_threshold:
                if self._checkpoint_waypoint_index == 0:
                    self._checkpoint_waypoint_index = self._next_checkpoint_waypoint_index
                    self._next_checkpoint_waypoint_index += self._checkpoint_frequency
                else:
                    self._checkpoint_waypoint_index = 0

                self._repeat_count = 0

    def _update_checkpoint_by_section(self):
        s1, s2, s3 = self.sections_indexes

        # s[0] is start and s[1] is end of section. s[2] is checkpoint frequency
        if s1[0] <= self._start_waypoint_index <= s1[1]:
            start = s1[0]
            end = s1[1]
            frequency = s1[2]
        elif s2[0] <= self._start_waypoint_index <= s2[1]:
            start = s2[0]
            end = s2[1]
            frequency = s2[2]
        else:
            # s3
            start = s3[0]
            end = s3[1]
            frequency = s3[2]

        if self._current_waypoint_index == end:
            idx = end
        else:
            # get the most recent progress in term of index complete
            # discard any progress that doesn't reach checkpoint index
            idx = (((self._current_waypoint_index - start) // frequency) * frequency) + start

        if idx >= self._next_checkpoint_waypoint_index:
            self._repeat_count += 1

            if self._repeat_count >= self._repeat_count_threshold:
                if self._checkpoint_waypoint_index == start:
                    if self._next_checkpoint_waypoint_index >= end:
                        self._checkpoint_waypoint_index, frequency = self._get_next_section_start_and_frequency(end)
                        self._next_checkpoint_waypoint_index = self._checkpoint_waypoint_index + frequency
                    else:
                        self._checkpoint_waypoint_index = self._next_checkpoint_waypoint_index
                        self._next_checkpoint_waypoint_index += frequency

                        self._next_checkpoint_waypoint_index = min(self._next_checkpoint_waypoint_index, end)
                else:
                    self._checkpoint_waypoint_index = start

                self._repeat_count = 0

    def _get_next_section_start_and_frequency(self, end_of_section):
        end_idx = self.sections_end.index(end_of_section)
        next_start = self.sections_start[(end_idx + 1) % len(self.sections_start)]
        next_frequency = self.sections_frequency[(end_idx + 1) % len(self.sections_frequency)]
        return next_start, next_frequency

    def _get_random_spawn_point(self):
        start_original = random.random() >= 0.4
        if start_original:
            self._in_random_spawn_point = False
            spawn_idx = self._checkpoint_waypoint_index
        else:
            self._in_random_spawn_point = True

            if random.random() >= 0.3 or self._checkpoint_waypoint_index in self.sections_start:
                # random start in the same section
                spawn_idx = self._checkpoint_waypoint_index + (random.randint(5, 20) // 2 * 2)
            else:
                # random start at any point before current checkpoint
                lower_bound = 0
                for start, end in zip(self.sections_start, self.sections_ends):
                    if start <= self._checkpoint_waypoint_index < end:
                        lower_bound = start
                        break

                spawn_idx = random.randint(lower_bound, self._checkpoint_waypoint_index)

        spawn_transform = _route_waypoints[spawn_idx][0].transform

        return spawn_idx, spawn_transform

    def _get_cycle_spawn_point(self):
        list_idx = self.round_spawn_idx % len(self.all_spawn_indexes)
        self.round_spawn_idx += 1

        spawn_idx = self.all_spawn_indexes[list_idx]
        spawn_transform = _route_waypoints[spawn_idx][0].transform

        return spawn_idx, spawn_transform

    def _get_AIT_spawn_point(self):
        return 0, _route_waypoints[0][0].transform

    def _is_AIT_map(self):
        return self._map.name == 'ait_v4/Maps/ait_v4/ait_v4'

    def _update_reached_last_waypoint_index(self, waypoint_index):
        if self._is_AIT_map():
            # TODO: implement this
            return False
        else:
            waypoint_routes_len = len(_route_waypoints)
            return (waypoint_index % waypoint_routes_len) == self.sections_ends[-1]

    def _get_waypoint_forward_vector_np(self, waypoint):
        forward_vector = waypoint.transform.get_forward_vector()
        vector_np = np.array([forward_vector.x, forward_vector.y, forward_vector.z])

        if self.traffic_mode == 'LHT':
            vector_np = vector_np * -1

        return vector_np

    def _draw_debug_waypoint(self, waypoint):
        self.carla_debug.draw_point(waypoint.transform.location, size=0.3, life_time=60)

    @property
    def next_waypoint(self):
        return _route_waypoints[(self._current_waypoint_index + 1) % len(_route_waypoints)][0]

    @property
    def current_waypoint(self):
        return _route_waypoints[self._current_waypoint_index % len(_route_waypoints)][0]

    @property
    def is_end_of_section(self):
        return self._current_waypoint_index in self.sections_ends


TOWN7_PLAN = [RoadOption.STRAIGHT] + [RoadOption.RIGHT] * 2 + [RoadOption.STRAIGHT] * 5
