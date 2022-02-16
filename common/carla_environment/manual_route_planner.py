# copy from https://github.com/bitsauce/Carla-ppo/blob/master/CarlaEnv/planner.py
import numpy as np

import carla
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import vector
from .misc import distance_vehicle

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


class ManualRoutePlanner:
    def __init__(self, start_waypoint, end_waypoint, resolution=2.0, plan=None):
        self._vehicle = None
        self._world = None
        self._map = None

        self._sampling_radius = resolution
        self._min_distance = self._sampling_radius - 1 if self._sampling_radius > 1 else 1

        self.start_waypoint = start_waypoint
        self.end_waypoint = end_waypoint
        self.spawn_transform = start_waypoint.transform
        self.lap_count = 0

        self._current_waypoint_index = 0
        self._checkpoint_waypoint_index = 0
        self._start_waypoint_index = 0
        self._checkpoint_frequency = 50

        self.plan = plan

    def set_vehicle(self, vehicle):
        ''' Set internal state to current vehicle, must be called in `reset` '''
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._start_waypoint_index = self._checkpoint_waypoint_index
        self.lap_count = 0

    def run_step(self):
        global _route_waypoints, _transformed_waypoint_routes

        if _route_waypoints is None:
            _route_waypoints = self._compute_route_waypoints()
            _transformed_waypoint_routes = self._transform_waypoints(_route_waypoints)

        waypoint_routes_len = len(_route_waypoints)
        current_transform = self._vehicle.get_transform()
        waypoint_index = self._current_waypoint_index
        for _ in range(waypoint_routes_len):
            # check if we passed next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = _route_waypoints[next_waypoint_index % waypoint_routes_len]
            dot = np.dot(carla_to_vector(wp.transform.get_forward_vector())[:2],
                         carla_to_vector(current_transform.location - wp.transform.location)[:2])

            # did we pass the waypoint?
            if dot > 0.0:
                # if passed, go to next waypoint
                waypoint_index += 1
                continue

            # distance must be greater than min distance
            if distance_vehicle(wp, current_transform) < self._min_distance:
                waypoint_index += 1
            else:
                break

        self._current_waypoint_index = waypoint_index

        # update checkpoint
        self._checkpoint_waypoint_index = (self._current_waypoint_index // self._checkpoint_frequency) * self._checkpoint_frequency
        self.spawn_transform = _route_waypoints[self._checkpoint_waypoint_index][0].transform

        self.lap_count = (self._current_waypoint_index - self._start_waypoint_index) / len(_route_waypoints)

        return _transformed_waypoint_routes[self._current_waypoint_index:]

    def get_route_waypoints(self):
        ''' Calculate and get waypoints along the route and return list of [waypoint, RoadOption] '''
        global _route_waypoints

        if _route_waypoints is None:
            _route_waypoints = self._compute_route_waypoints()

        return _route_waypoints

    def _compute_route_waypoints(self):
        """
            Returns a list of (waypoint, RoadOption)-tuples that describes a route
            starting at start_waypoint, ending at end_waypoint.

            start_waypoint (carla.Waypoint):
                Starting waypoint of the route
            end_waypoint (carla.Waypoint):
                Destination waypoint of the route
            resolution (float):
                Resolution, or lenght, of the steps between waypoints
                (in meters)
            plan (list(RoadOption) or None):
                If plan is not None, generate a route that takes every option as provided
                in the list for every intersections, in the given order.
                (E.g. set plan=[RoadOption.STRAIGHT, RoadOption.LEFT, RoadOption.RIGHT]
                to make the route go straight, then left, then right.)
                If plan is None, we use the GlobalRoutePlanner to find a path between
                start_waypoint and end_waypoint.
        """

        if self.plan is None:
            # Setting up global router
            grp = GlobalRoutePlanner(self._map, self._sampling_radius)
            
            # Obtain route plan
            route = grp.trace_route(
                self.start_waypoint.transform.location,
                self.end_waypoint.transform.location)
        else:
            # Compute route waypoints
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
        return list(map(lambda wp: [wp[0].transform.location.x, wp[0].transform.location.y, wp[0].transform.rotation.yaw], waypoints))


TOWN4_PLAN = [RoadOption.STRAIGHT] + [RoadOption.RIGHT] * 2 + [RoadOption.STRAIGHT] * 5
TOWN4_REVERSE_PLAN = [RoadOption.STRAIGHT] * 4 + [RoadOption.LEFT] * 2 + [RoadOption.STRAIGHT]
