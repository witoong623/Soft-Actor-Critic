import functools
import operator

from agents.navigation.local_planner import RoadOption


class AITRoutePlanner:
    def __init__(self, world, sampling_radius) -> None:
        self.world = world
        self.map = world.get_map()
        self._sampling_radius = sampling_radius

        self._route_waypoints = None
        self._route_transforms = None

        self._setup()

    def compute_route_waypoints(self):
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

    def get_compatible_route_waypoint(self):
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

