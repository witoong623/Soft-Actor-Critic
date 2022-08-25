import sys
sys.path.append('/home/witoon/thesis/code/Soft-Actor-Critic')

import carla

from common.carla_environment.route_tracker import RouteTracker
from common.carla_environment.custom_route_planner import AITRoutePlanner
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner


host = 'localhost'
port = 2000
map = 'ait_v4'
dt = 0.1
debug_life_time = 120

client = carla.Client(host, port)
client.set_timeout(30.0)

world = client.load_world(map)
# world = client.get_world()
debug = world.debug

color_white = carla.Color(r=255, g=255, b=255)
green = carla.Color(r=0, g=255, b=0)
red = carla.Color(r=255, g=0, b=0)


def draw_debug_transform(transform):
    draw_debug_location(transform.location)


def draw_debug_waypoint(waypoint, color=green):
    draw_debug_location(waypoint.transform.location, color=color)


def draw_forward_vector(waypoint):
    forward_vector = waypoint.get_forward_vector()
    begin = waypoint.location
    end = waypoint.location + (forward_vector * -1)

    debug.draw_arrow(begin, end, thickness=0.2, arrow_size=0.2, life_time=debug_life_time)


def draw_debug_location(location, color=green):
    debug.draw_point(location, size=0.3, life_time=debug_life_time, color=color)


def draw_debug_text_location(location, text):
    debug.draw_string(location, text=text, life_time=debug_life_time, color=color_white)


if __name__ == '__main__':
    ait_map = world.get_map()

    route_planner = AITRoutePlanner(world, 2)
    route_transforms = route_planner.compute_route()
    route_waypoints = route_planner.get_carla_agent_compatible_route_waypoint()

    print('len is', len(route_transforms))

    # for i, (transform, _) in enumerate(route_transforms):
    #     draw_debug_location(transform)
    #     # draw_forward_vector(waypoint)
    #     draw_debug_text_location(transform.location, text=str(i))

    for i, (waypoint, _) in enumerate(route_waypoints):
        color = green
        if waypoint.is_junction:
            color = red

        draw_debug_waypoint(waypoint, color)
        # draw_forward_vector(waypoint)
        draw_debug_text_location(waypoint.transform.location, text=str(i))

    # route_waypoints = route_planner.get_carla_agent_compatible_route_waypoint()

    # junction_start = 316
    # junction_end = 322
    # for wp, _ in route_waypoints[junction_start:junction_end]:
    #     print('is waypoint junction', wp.is_junction)
