import sys
sys.path.append('/home/witoon/thesis/code/Soft-Actor-Critic')

import carla

from common.carla_environment.route_tracker import RouteTracker, TOWN7_PLAN



host = 'localhost'
port = 2000
map = 'Town07'
dt = 0.1

client = carla.Client(host, port)
client.set_timeout(30.0)

world = client.load_world(map)

settings = world.get_settings()
# settings.fixed_delta_seconds = dt
# settings.synchronous_mode = True
# world.apply_settings(settings)

vehicle_spawn_points = list(world.get_map().get_spawn_points())
lap_spwan_point_wp = world.get_map().get_waypoint(vehicle_spawn_points[1].location)

route_tracker = RouteTracker(lap_spwan_point_wp, lap_spwan_point_wp, resolution=2, plan=TOWN7_PLAN)
print(f'current wp after planner created is {route_tracker._current_waypoint_index}')

bp_library = world.get_blueprint_library()

ego_bp = bp_library.find('vehicle.nissan.micra')

ego_spwan_transform = route_tracker.spawn_transform
ego_vehicle = world.try_spawn_actor(ego_bp, ego_spwan_transform)

route_tracker.set_vehicle(ego_vehicle)

# waypoints = routeplanner.get_transformed_route_waypoints()
# waypoints = routeplanner.run_step()
waypoints = route_tracker.get_route_waypoints()

print(f'route waypoint len is {len(waypoints)}')

debug = world.debug
green = carla.Color(r=0, g=255, b=0)
red = carla.Color(r=255, g=0, b=0)
life_time = 120
for i, (wp, action) in enumerate(waypoints):
    location = wp.transform.location
    location.z = 0.1
    debug.draw_point(location, size=0.3, life_time=life_time, color=red)
    location.y = location.y - 1
    debug.draw_string(location, text=f'{i}', life_time=life_time, color=green)

# for i, wp in enumerate(waypoints[125:150], 125):
#     used_color = green
#     # if i == 125:
#     #     used_color = red
#     transform = wp[0].transform
#     location = transform.location
#     rotation = transform.rotation
#     debug.draw_string(location, text=f'{i} - :x: {location.x}, y: {location.y}, z: {location.z}, yaw: {rotation.yaw}', life_time=60)

world.wait_for_tick()
