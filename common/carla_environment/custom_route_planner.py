import abc
import functools
import operator

import carla
import numpy as np

from agents.navigation.local_planner import RoadOption
from agents.tools.misc import vector


class BaseRoutePlanner(abc.ABC):
    def __init__(self, world, sampling_radius):
        self.world = world
        self.map = world.get_map()
        self._sampling_radius = sampling_radius

        self._route_waypoints = None
        self._route_transforms = None

    @abc.abstractclassmethod
    def compute_route(self):
        pass

    @abc.abstractclassmethod
    def get_compatible_route_waypoint(self):
        pass


class Town7RoutePlanner(BaseRoutePlanner):
    def __init__(self, world, sampling_radius, start_waypoint, end_waypoint, plan):
        super().__init__(world, sampling_radius)

        self._plan = plan
        self.start_waypoint = start_waypoint
        self.end_waypoint = end_waypoint

    def compute_route(self):
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

        self._route_waypoints = route
        self._route_transforms = [(wp.transform, action) for wp, action in self._route_waypoints]

        return route


class AITRoutePlanner(BaseRoutePlanner):
    def __init__(self, world, sampling_radius):
        super().__init__(world, sampling_radius)

        self._setup()

    def compute_route(self):
        if self._route_waypoints is not None:
            return self._route_waypoints

        lap_route = []

        first_start = self._get_waypoint_opposite_of(11)
        lap_route.append(self._get_route_until_end(first_start, 'previous', choice_index=2))

        second_start = self._get_waypoint_opposite_of(9)
        lap_route.append(self._get_route_until_end(second_start, 'previous', choice_index=1))

        third_start = self._get_waypoint_opposite_of(5)
        lap_route.append(self._get_route_until_end(third_start, 'previous'))

        fourth_start = self._get_waypoint_at(34)
        lap_route.append(self._get_route_until_end(fourth_start, 'next'))

        fifth_start = fourth_start
        lap_route.append(self._get_route_until_end(fifth_start, 'previous', choice_index=1))

        sixth_start = self._get_waypoint_opposite_of(49)
        lap_route.append(self._get_route_until_end(sixth_start, 'previous'))

        seventh_start = self._get_waypoint_at(21)
        lap_route.append(self._get_route_until_end(seventh_start, 'next'))

        eighth_start = self._get_waypoint_at(46)
        lap_route.append(self._get_route_until_end(eighth_start, 'next'))

        self._route_waypoints = list(functools.reduce(operator.concat, lap_route))
        self._route_transforms = [wp.transform for wp in self._route_waypoints]

        return self._add_compatibility_support(self._route_transforms)

    def get_carla_agent_compatible_route_waypoint(self):
        return [(waypoint, RoadOption.LANEFOLLOW) for waypoint in self._route_waypoints]

    def _get_route_until_end(self, start_waypoint, next_func, choice_index=None):
        route = []
        current_waypoint = start_waypoint
        choices = [start_waypoint]

        while len(choices) > 0:
            if next_func == 'next':
                choices = current_waypoint.next(self._sampling_radius)
            else:
                choices = current_waypoint.previous(self._sampling_radius)

            if len(choices) > 1:
                assert choice_index is not None

                current_waypoint = choices[choice_index]
            elif len(choices) == 1:
                current_waypoint = choices[0]
            else:
                continue
            
            route.append(current_waypoint)

        if next_func == 'next':
            route.reverse()

        return route


    def _get_waypoint_opposite_of(self, spawn_point_index):
        waypoint = self.map.get_waypoint(self.spawn_points[spawn_point_index].location)
        next_waypoints = waypoint.next(self._sampling_radius)

        assert len(next_waypoints) > 0
        return next_waypoints[0].get_left_lane()

    def _get_waypoint_at(self, spawn_point_index):
        return self.map.get_waypoint(self.spawn_points[spawn_point_index].location)

    def _setup(self):
        self.spawn_points = list(self.map.get_spawn_points())

    def _add_compatibility_support(self, route_objects):
        ''' Add arbitrary road option to route object (Waypoint or Transform) '''
        return [(route_obj, RoadOption.LANEFOLLOW) for route_obj in route_objects]

