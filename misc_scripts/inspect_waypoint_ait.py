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

# world = client.load_world(map)
world = client.get_world()
debug = world.debug

color_white = carla.Color(r=255, g=255, b=255)
green = carla.Color(r=0, g=255, b=0)
red = carla.Color(r=255, g=0, b=0)


def draw_debug_transform(waypoint):
    draw_debug_location(waypoint.location)


def draw_forward_vector(waypoint):
    forward_vector = waypoint.get_forward_vector()
    begin = waypoint.location
    end = waypoint.location + (forward_vector * -1)

    debug.draw_arrow(begin, end, thickness=0.2, arrow_size=0.2, life_time=debug_life_time)


def draw_debug_location(location):
    debug.draw_point(location, size=0.3, life_time=debug_life_time)


def draw_debug_text_location(location, text):
    debug.draw_string(location, text=text, life_time=debug_life_time, color=color_white)


if __name__ == '__main__':
    ait_map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    # libraries = blueprint_library.filter('vehicle.*.*')

    # print(libraries)

    route_planner = AITRoutePlanner(world, 2)

    route_waypoints = route_planner.compute_route()

    print('len is', len(route_waypoints))

    for i, (transform, _) in enumerate(route_waypoints):
        draw_debug_transform(transform)
        # draw_forward_vector(waypoint)
        draw_debug_text_location(transform.location, text=str(i))

    # vehicle_spawn_points = list(ait_map.get_spawn_points())

    # for i, spawn_point in enumerate(vehicle_spawn_points):
    #     draw_debug_location(spawn_point.location)
    #     draw_debug_text_location(spawn_point.location, text=str(i))

    # spawn_point_46 = vehicle_spawn_points[46]
    # print(type(spawn_point_46))
    # spawn_point_34 = vehicle_spawn_points[34]
    # spawn_point_52 = vehicle_spawn_points[52]
    # spawn_point_56 = vehicle_spawn_points[56]

    # spawn_point_11 = ait_map.get_waypoint(vehicle_spawn_points[11].location)
    # spawn_point_11_next = spawn_point_11.next(2)
    # final_destination_location = spawn_point_11_next[0].get_left_lane().transform.location
    # draw_debug_location(final_destination_location)

    # global_route_planner = GlobalRoutePlanner(ait_map, 2)
    # route = global_route_planner.trace_route(spawn_point_46.location, spawn_point_34.location)

    # routeplanner = ManualRoutePlanner(starting_waypoint, starting_waypoint, world=world,
    #                                 resolution=2, plan=None, traffic_mode='LHT')

    # waypoints = routeplanner.get_route_waypoints()

    # print(f'route waypoint len is {len(waypoints)}')


    # bp_library = world.get_blueprint_library()

    # ego_bp = bp_library.find('vehicle.nissan.micra')

    # ego_spwan_transform = routeplanner.spawn_transform
    # ego_vehicle = world.try_spawn_actor(ego_bp, ego_spwan_transform)

    # routeplanner.set_vehicle(ego_vehicle)

    # # waypoints = routeplanner.get_transformed_route_waypoints()
    # # waypoints = routeplanner.run_step()
    # waypoints = routeplanner.get_route_waypoints()

    # print(f'route waypoint len is {len(waypoints)}')


    # life_time = 120
    # for i, (wp, action) in enumerate(waypoints):
    #     location = wp.transform.location
    #     location.z = 0.1
    #     debug.draw_point(location, size=0.3, life_time=life_time, color=red)
    #     location.y = location.y - 1
    #     debug.draw_string(location, text=f'{i}', life_time=life_time, color=green)

    # for i, wp in enumerate(waypoints[125:150], 125):
    #     used_color = green
    #     # if i == 125:
    #     #     used_color = red
    #     transform = wp[0].transform
    #     location = transform.location
    #     rotation = transform.rotation
    #     debug.draw_string(location, text=f'{i} - :x: {location.x}, y: {location.y}, z: {location.z}, yaw: {rotation.yaw}', life_time=60)
